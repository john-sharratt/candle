//! What the slot fill weighed each working piece at, and what it placed.
//!
//! The fill ranks a working piece's rounds by each turn's own score
//! (`dialogue_fill::plan`), while the projection's selection record stamps
//! every member turn with its piece's momentum: read from the record, every
//! round of a piece looks alike and the fill's reasons are invisible. This
//! reports, per piece, the scores the fill actually ranked by — the best of
//! its turns, the best it left out — beside what it placed and where the
//! piece stands in the pool, and the strongest pieces the scan scored that
//! the pool did not take. Each projection record carries it
//! (`ProjectionEvent::pieces`, `ProjectionEvent::unpooled`).

use serde::{Deserialize, Serialize};

use super::dialogue_fill::WorkingPiece;
use super::ids::{TimelineId, TurnKey};

/// Where one conversation stands in the dialogue's pool
/// (`docs/hit_driven_projection.md` §7.1): its momentum, the scan's fresh
/// score that fed it this boundary, its hit momentum, and the strength the
/// pool ranks by.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Standing {
    pub momentum: f32,
    pub fresh: f32,
    pub hits: f32,
    pub strength: f32,
    /// The file or folder the conversation reads, when it reads one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// One working piece as the fill saw it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PieceReport {
    pub timeline: u64,
    /// The cap it fills under: `Reading`, `Folder`, `Guide` or `Journal`.
    pub kind: String,
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
    #[serde(flatten)]
    pub standing: Standing,
}

/// A conversation the scan scored this boundary that the pool does not hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unpooled {
    pub timeline: u64,
    #[serde(flatten)]
    pub standing: Standing,
}

/// How many unpooled conversations a record keeps, strongest fresh score
/// first.
pub const UNPOOLED_KEPT: usize = 24;

/// Each piece's report, strongest first by its best turn.
pub fn piece_reports(
    working: &[WorkingPiece],
    placed: &[Vec<TurnKey>],
    score: impl Fn(TurnKey) -> f32,
    tokens: impl Fn(TurnKey) -> usize,
    standing: impl Fn(TimelineId) -> Standing,
) -> Vec<PieceReport> {
    let mut out: Vec<PieceReport> = working
        .iter()
        .zip(placed)
        .filter_map(|(piece, placed)| {
            let turns: Vec<TurnKey> = piece.exchanges.iter().flatten().copied().collect();
            let timeline = turns.first()?.timeline;
            let best = turns.iter().map(|&k| score(k)).fold(0.0, f32::max);
            let best_left = turns
                .iter()
                .filter(|k| !placed.contains(k))
                .map(|&k| score(k))
                .fold(0.0, f32::max);
            Some(PieceReport {
                timeline: timeline.raw(),
                kind: format!("{:?}", piece.kind),
                best,
                best_left,
                scored: turns.iter().filter(|&&k| score(k) > 0.0).count(),
                turns: turns.len(),
                placed: placed.len(),
                placed_tokens: placed.iter().map(|&k| tokens(k)).sum(),
                standing: standing(timeline),
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
                "{}:{} best={:.0} left={:.0} scored={}/{} placed={} tok={} fresh={:.0} m={:.0} h={:.2}",
                r.timeline,
                r.kind,
                r.best,
                r.best_left,
                r.scored,
                r.turns,
                r.placed,
                r.placed_tokens,
                r.standing.fresh,
                r.standing.momentum,
                r.standing.hits,
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::projection::fill::Kind;
    use crate::projection::TurnIndex;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn key(raw: u64, i: u32) -> TurnKey {
        TurnKey::new(tl(raw), TurnIndex(i))
    }

    /// **Each piece is reported at the scores the fill ranks by**: its best
    /// turn, its best left out, how many turns scored, what it placed, and
    /// where it stands in the pool — strongest first.
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
        let standing = |t: TimelineId| Standing {
            momentum: 10.0 * t.raw() as f32,
            fresh: 5.0,
            hits: 0.5,
            strength: 15.0 * t.raw() as f32,
            path: Some(format!("f{}.rs", t.raw())),
        };
        let reports = piece_reports(&working, &placed, score, |_| 100, standing);
        assert_eq!(
            reports,
            vec![
                PieceReport {
                    timeline: 2,
                    kind: "Reading".into(),
                    best: 1_200.0,
                    best_left: 1_200.0,
                    scored: 1,
                    turns: 2,
                    placed: 0,
                    placed_tokens: 0,
                    standing: Standing {
                        momentum: 20.0,
                        fresh: 5.0,
                        hits: 0.5,
                        strength: 30.0,
                        path: Some("f2.rs".into()),
                    },
                },
                PieceReport {
                    timeline: 1,
                    kind: "Reading".into(),
                    best: 900.0,
                    best_left: 900.0,
                    scored: 2,
                    turns: 3,
                    placed: 2,
                    placed_tokens: 200,
                    standing: Standing {
                        momentum: 10.0,
                        fresh: 5.0,
                        hits: 0.5,
                        strength: 15.0,
                        path: Some("f1.rs".into()),
                    },
                },
            ]
        );
        assert_eq!(
            line(&reports, 1),
            "2:Reading best=1200 left=1200 scored=1/2 placed=0 tok=0 fresh=5 m=20 h=0.50"
        );
    }

    /// **A record names a piece by its path and keeps its standing flat**,
    /// beside the fill's own fields.
    #[test]
    fn a_report_serializes_flat() {
        let report = PieceReport {
            timeline: 7,
            kind: "Reading".into(),
            best: 1.0,
            best_left: 0.0,
            scored: 1,
            turns: 1,
            placed: 1,
            placed_tokens: 3,
            standing: Standing {
                momentum: 2.0,
                fresh: 4.0,
                hits: 0.0,
                strength: 2.0,
                path: Some("a.rs".into()),
            },
        };
        assert_eq!(
            serde_json::to_string(&report).unwrap(),
            r#"{"timeline":7,"kind":"Reading","best":1.0,"best_left":0.0,"scored":1,"turns":1,"placed":1,"placed_tokens":3,"momentum":2.0,"fresh":4.0,"hits":0.0,"strength":2.0,"path":"a.rs"}"#
        );
    }
}
