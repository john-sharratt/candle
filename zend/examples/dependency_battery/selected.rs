//! P1's files: what provenance selection placed when C1 asked the question.
//!
//! P1 asks each question twice. The first ask is C1's — the production
//! memory, its dials and its prompt — and is read only for its projection
//! points: the point governing the answer's first token names the readings it
//! placed, strongest first (`candle_conversation::projection::fill_report`).
//! Those files, read whole from the repository, are packed into P1's prompt in
//! that order exactly as R1's BM25 files are in theirs (`rag::context::pack`,
//! the same budget). The second ask is then R1's in every respect but the
//! files, so P1 against R1 compares provenance's ranking of files with BM25's,
//! both read as text.

use crate::placement::{answer_token, Point};
use crate::view::governing;

/// The repository-relative paths of the readings the point governing the
/// answer's first token placed, in the point's order — strongest first — and
/// each once. A reading's path is workspace-relative, its repository first;
/// one outside `repo` — another repository's file — is left out.
pub fn selected(points: &[Point], think: Option<u32>, repo: &str) -> Vec<String> {
    let starts: Vec<u32> = points.iter().map(|p| p.start_token).collect();
    let Some(at) = governing(&starts, answer_token(think)) else {
        return Vec::new();
    };
    let prefix = format!("{repo}/");
    let placed = points[at]
        .placed
        .iter()
        .filter(|p| p.kind == "Reading")
        .filter_map(|p| p.path.as_deref()?.strip_prefix(&prefix));
    let mut out: Vec<String> = Vec::new();
    for path in placed {
        if !out.iter().any(|p| p == path) {
            out.push(path.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::Piece;

    fn piece(kind: &str, path: &str) -> Piece {
        Piece {
            kind: kind.into(),
            path: Some(path.into()),
        }
    }

    /// **The answer's own point names the files**, in its own order: the
    /// opening's placements are not the answer's, a folder or another
    /// repository's file is no reading of this one, and a file placed twice is
    /// listed once, where it first ranks.
    #[test]
    fn the_answers_point_names_its_readings_in_its_order() {
        let points = vec![
            Point {
                start_token: 0,
                placed: vec![piece("Reading", "candle/early.rs")],
            },
            Point {
                start_token: 200,
                placed: vec![
                    piece("Reading", "candle/a.rs"),
                    piece("Folder", "candle/zend/"),
                    piece("Reading", "journals/x.yaml"),
                    piece("Reading", "candle/b.rs"),
                    piece("Reading", "candle/a.rs"),
                ],
            },
        ];
        assert_eq!(
            selected(&points, Some(250), "candle"),
            vec!["a.rs".to_string(), "b.rs".to_string()]
        );
        assert_eq!(selected(&points, None, "candle"), vec!["early.rs"]);
        assert!(selected(&[], Some(10), "candle").is_empty());
    }
}
