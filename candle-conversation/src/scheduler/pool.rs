//! Which pieces a dialogue's pool holds (`docs/hit_driven_projection.md` §7.2).
//!
//! The pool stands behind the slot: whole conversations — a file's reading, a
//! guide, the dialogue itself — whose index rows are resident at the sampled
//! layers, so the shadow scan can say which of their rounds the indexer would
//! have read. It admits by piece, never part of one: the pieces holding a
//! floor first — every piece the slot places — then the rest by score,
//! provenance's base lifted by the piece's hit momentum, until it holds
//! `pool_tokens`.

use std::collections::HashMap;

use crate::projection::TimelineId;

/// The `pool_tokens` a card of `total_bytes` of device memory holds when its
/// machine config names none: 1M tokens on the 72 GB card, 512K on the 24 GB
/// card, 256K on the 16 GB card.
pub fn default_pool_tokens(total_bytes: u64) -> usize {
    const GIB: u64 = 1 << 30;
    match total_bytes {
        b if b >= 48 * GIB => 1_000_000,
        b if b >= 20 * GIB => 512_000,
        _ => 256_000,
    }
}

/// One candidate piece: its conversation, its tokens, and its score —
/// provenance's base already lifted by its hits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoolCandidate {
    pub piece: TimelineId,
    pub tokens: usize,
    pub score: f32,
}

/// The pieces the pool holds: every one of `floors` whatever it costs, then
/// `candidates` by score, highest first, each whole piece while it fits in
/// what `pool_tokens` leaves. A piece that does not fit is passed over for a
/// smaller one behind it. Ties go to the earlier candidate.
pub fn choose_pool(
    floors: &[(TimelineId, usize)],
    candidates: &[PoolCandidate],
    pool_tokens: usize,
) -> Vec<TimelineId> {
    let mut held: HashMap<TimelineId, ()> = HashMap::new();
    let mut out = Vec::new();
    let mut used = 0usize;
    for &(piece, tokens) in floors {
        if held.insert(piece, ()).is_none() {
            used += tokens;
            out.push(piece);
        }
    }
    let mut ranked: Vec<(usize, &PoolCandidate)> = candidates.iter().enumerate().collect();
    ranked.sort_by(|(ia, a), (ib, b)| b.score.total_cmp(&a.score).then(ia.cmp(ib)));
    for (_, c) in ranked {
        if held.contains_key(&c.piece) || used + c.tokens > pool_tokens {
            continue;
        }
        held.insert(c.piece, ());
        used += c.tokens;
        out.push(c.piece);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn c(piece: u64, tokens: usize, score: f32) -> PoolCandidate {
        PoolCandidate {
            piece: tl(piece),
            tokens,
            score,
        }
    }

    #[test]
    fn each_card_has_its_pool() {
        assert_eq!(default_pool_tokens(73_415 << 20), 1_000_000);
        assert_eq!(default_pool_tokens(24_576 << 20), 512_000);
        assert_eq!(default_pool_tokens(16_376 << 20), 256_000);
    }

    /// Floors first whatever they cost, then by score while each whole piece
    /// fits; one that does not fit gives way to a smaller one behind it.
    #[test]
    fn floors_then_score_while_whole_pieces_fit() {
        let floors = [(tl(1), 300)];
        let cands = [
            c(2, 500, 0.9),
            c(3, 200, 0.8),
            c(4, 100, 0.5),
            c(1, 300, 0.99),
        ];
        assert_eq!(
            choose_pool(&floors, &cands, 700),
            vec![tl(1), tl(3), tl(4)],
            "piece 2 does not fit beside the floor; 3 and 4 do"
        );
        assert_eq!(
            choose_pool(&floors, &cands, 1_100),
            vec![tl(1), tl(2), tl(3)]
        );
        // The floors stand even past the pool.
        assert_eq!(choose_pool(&floors, &cands, 100), vec![tl(1)]);
    }

    /// Equal scores keep the candidates' own order.
    #[test]
    fn ties_go_to_the_earlier_candidate() {
        let cands = [c(5, 10, 0.5), c(6, 10, 0.5), c(7, 10, 0.5)];
        assert_eq!(choose_pool(&[], &cands, 20), vec![tl(5), tl(6)]);
    }
}
