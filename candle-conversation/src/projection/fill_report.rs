//! What the slot fill weighed each working piece at, and what it placed.
//!
//! The fill ranks a working piece's rounds by each turn's own score
//! (`dialogue_fill::plan`), while the projection's selection record stamps
//! every member turn with its piece's momentum: read from the record, every
//! round of a piece looks alike and the fill's reasons are invisible. This
//! reports, per piece, the scores the fill actually ranked by — the best of
//! its turns, the best it left out — beside what it placed.

use super::dialogue_fill::WorkingPiece;
use super::fill::Kind;
use super::ids::TurnKey;

/// One working piece as the fill saw it.
#[derive(Debug, Clone, PartialEq)]
pub struct PieceReport {
    pub timeline: u64,
    pub kind: Kind,
    /// Its best turn's score.
    pub best: f32,
    /// Its best score among the turns the fill did not place; `0` when it
    /// placed them all.
    pub best_left: f32,
    /// Turns with a score above zero.
    pub scored: usize,
    pub turns: usize,
    pub placed: usize,
    pub placed_tokens: usize,
}

/// Each piece's report, strongest first by its best turn.
pub fn piece_reports(
    working: &[WorkingPiece],
    placed: &[Vec<TurnKey>],
    score: impl Fn(TurnKey) -> f32,
    tokens: impl Fn(TurnKey) -> usize,
) -> Vec<PieceReport> {
    let mut out: Vec<PieceReport> = working
        .iter()
        .zip(placed)
        .filter_map(|(piece, placed)| {
            let turns: Vec<TurnKey> = piece.exchanges.iter().flatten().copied().collect();
            let timeline = turns.first()?.timeline.raw();
            let best = turns.iter().map(|&k| score(k)).fold(0.0, f32::max);
            let best_left = turns
                .iter()
                .filter(|k| !placed.contains(k))
                .map(|&k| score(k))
                .fold(0.0, f32::max);
            Some(PieceReport {
                timeline,
                kind: piece.kind,
                best,
                best_left,
                scored: turns.iter().filter(|&&k| score(k) > 0.0).count(),
                turns: turns.len(),
                placed: placed.len(),
                placed_tokens: placed.iter().map(|&k| tokens(k)).sum(),
            })
        })
        .collect();
    out.sort_by(|a, b| b.best.total_cmp(&a.best));
    out
}

/// The `n` strongest reports as one line.
pub fn line(reports: &[PieceReport], n: usize) -> String {
    reports
        .iter()
        .take(n)
        .map(|r| {
            format!(
                "{}:{:?} best={:.0} left={:.0} scored={}/{} placed={} tok={}",
                r.timeline,
                r.kind,
                r.best,
                r.best_left,
                r.scored,
                r.turns,
                r.placed,
                r.placed_tokens
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::projection::{TimelineId, TurnIndex};

    fn key(tl: u64, i: u32) -> TurnKey {
        TurnKey::new(TimelineId::from_raw(tl).unwrap(), TurnIndex(i))
    }

    /// **Each piece is reported at the scores the fill ranks by**: its best
    /// turn, its best left out, how many turns scored, and what it placed —
    /// strongest first.
    #[test]
    fn a_piece_is_reported_at_its_turns_scores() {
        let working = vec![
            WorkingPiece {
                kind: Kind::Reading,
                exchanges: vec![vec![key(1, 0), key(1, 1), key(1, 2)]],
            },
            WorkingPiece {
                kind: Kind::Reading,
                exchanges: vec![vec![key(2, 0), key(2, 1)]],
            },
        ];
        let placed = vec![vec![key(1, 0), key(1, 2)], vec![]];
        let score = |k: TurnKey| match (k.timeline.raw(), k.index.0) {
            (1, 0) => 400.0,
            (1, 1) => 900.0,
            (1, 2) => 0.0,
            (2, 0) => 1_200.0,
            _ => 0.0,
        };
        let reports = piece_reports(&working, &placed, score, |_| 100);
        assert_eq!(
            reports,
            vec![
                PieceReport {
                    timeline: 2,
                    kind: Kind::Reading,
                    best: 1_200.0,
                    best_left: 1_200.0,
                    scored: 1,
                    turns: 2,
                    placed: 0,
                    placed_tokens: 0,
                },
                PieceReport {
                    timeline: 1,
                    kind: Kind::Reading,
                    best: 900.0,
                    best_left: 900.0,
                    scored: 2,
                    turns: 3,
                    placed: 2,
                    placed_tokens: 200,
                },
            ]
        );
        assert_eq!(
            line(&reports, 1),
            "2:Reading best=1200 left=1200 scored=1/2 placed=0 tok=0"
        );
    }
}
