//! Whether an answer's file was in what the model saw, read from its turn's
//! projection points (`GET /v1/conversations/{id}/placements`): the readings
//! §6.5's table and Figures 9.12-5 and 9.12-6 are drawn from.
//!
//! A point governs the decode from its `start_token` until the next point. A
//! file is *placed at* token `t` when the point governing `t` placed turns of
//! a reading of it. The turn's think block spans tokens `0..think`, so the
//! reasoning's last token is `think − 1` and the answer's first is `think`;
//! the reprojection queued on the committed `</think>` is recorded one token
//! on, so the answer's first token is read at `think + 1`.

use serde::{Deserialize, Serialize};

/// One projection point as the placements route answers it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Point {
    pub start_token: u32,
    #[serde(default)]
    pub placed: Vec<Piece>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Piece {
    pub kind: String,
    #[serde(default)]
    pub path: Option<String>,
}

/// What one answer's projection points say of its answer files.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    /// The points recorded for the turn.
    pub points: usize,
    /// Points selected inside the think block, after the opening.
    pub in_thinking: usize,
    /// An answer file placed by the opening projection.
    pub opening: bool,
    /// An answer file placed when the reasoning ended.
    pub think_end: bool,
    /// An answer file placed for the answer's first token.
    pub first_answer: bool,
    /// An answer file placed at any point of the turn.
    pub ever: bool,
    /// Each point's position and whether it placed an answer file, in order.
    pub series: Vec<(u32, bool)>,
}

/// Whether a placed piece's `path` — workspace-relative, its repository
/// first — is the repository file `file`.
pub fn is_file(path: &str, file: &str) -> bool {
    path == file || path.ends_with(&format!("/{file}"))
}

/// Whether `point` placed a reading of any of `files`.
pub fn places(point: &Point, files: &[String]) -> bool {
    point.placed.iter().any(|p| {
        p.kind == "Reading"
            && p.path
                .as_deref()
                .is_some_and(|path| files.iter().any(|f| is_file(path, f)))
    })
}

/// Whether the point of `series` governing token `t` placed an answer file.
pub fn placed_at(series: &[(u32, bool)], t: u32) -> bool {
    series
        .iter()
        .take_while(|(start, _)| *start <= t)
        .last()
        .is_some_and(|(_, placed)| *placed)
}

/// The reading of `points` for `files`, the think block `think` tokens long
/// (`None` for a turn that did not reason).
pub fn placement(points: &[Point], files: &[String], think: Option<u32>) -> Placement {
    let series: Vec<(u32, bool)> = points
        .iter()
        .map(|p| (p.start_token, places(p, files)))
        .collect();
    let at = |t: u32| placed_at(&series, t);
    let think = think.unwrap_or(0);
    Placement {
        points: points.len(),
        in_thinking: series.iter().filter(|(s, _)| *s > 0 && *s < think).count(),
        opening: at(0),
        think_end: at(think.saturating_sub(1)),
        first_answer: at(think + 1),
        ever: series.iter().any(|(_, p)| *p),
        series,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(start: u32, paths: &[&str]) -> Point {
        Point {
            start_token: start,
            placed: paths
                .iter()
                .map(|p| Piece {
                    kind: "Reading".into(),
                    path: Some(p.to_string()),
                })
                .collect(),
        }
    }

    #[test]
    fn a_path_is_the_file_with_or_without_its_repository() {
        assert!(is_file("candle/zend/src/a.rs", "zend/src/a.rs"));
        assert!(is_file("zend/src/a.rs", "zend/src/a.rs"));
        assert!(!is_file("candle/zend/src/ba.rs", "zend/src/a.rs"));
    }

    /// **The reasoning brings the file in**: missed at the opening, placed by
    /// the reprojection at 256 inside a 300-token think block, so held at the
    /// think block's end and at the answer's first token.
    #[test]
    fn a_file_reached_while_reasoning_is_read_at_each_mark() {
        let files = vec!["zend/src/a.rs".to_string()];
        let points = [
            point(0, &["candle/zend/src/other.rs"]),
            point(128, &[]),
            point(256, &["candle/zend/src/a.rs"]),
            point(301, &["candle/zend/src/a.rs"]),
        ];
        let p = placement(&points, &files, Some(300));
        assert_eq!(
            p,
            Placement {
                points: 4,
                in_thinking: 2,
                opening: false,
                think_end: true,
                first_answer: true,
                ever: true,
                series: vec![(0, false), (128, false), (256, true), (301, true)],
            }
        );
    }

    /// **Held, the opening's working set stands to the end of reasoning**,
    /// and only the `</think>` reprojection reaches the file.
    #[test]
    fn a_held_turn_places_the_file_only_after_its_reasoning() {
        let files = vec!["zend/src/a.rs".to_string()];
        let points = [point(0, &[]), point(301, &["candle/zend/src/a.rs"])];
        let p = placement(&points, &files, Some(300));
        assert_eq!(
            (p.in_thinking, p.opening, p.think_end, p.first_answer),
            (0, false, false, true)
        );
    }

    /// A folder or a journal is no reading of the file, whatever it names.
    #[test]
    fn only_a_reading_places_a_file() {
        let files = vec!["zend/src/a.rs".to_string()];
        let folder = Point {
            start_token: 0,
            placed: vec![Piece {
                kind: "Folder".into(),
                path: Some("candle/zend/src/a.rs".into()),
            }],
        };
        assert!(!places(&folder, &files));
        assert_eq!(placement(&[], &files, None), Placement::default());
    }
}
