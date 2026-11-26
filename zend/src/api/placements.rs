//! `GET /v1/conversations/{id}/placements` — what each recorded projection
//! point placed, by file.
//!
//! For every assistant turn of the conversation, in order, each of its
//! projection points: the decoded-token position it was selected at and the
//! working pieces its slot fill placed turns of — each by its kind and the
//! file or folder it reads. This is the projection record cut to the question
//! "was this file in what the model saw at this point", which the full point
//! (`/projections/{turn}/{event}`) answers only beside every placed turn's body.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::HeaderMap,
    response::Response,
};
use candle_conversation::files::FileKey;
use candle_conversation::projection::ProjectionEvent;
use serde::Serialize;

use super::compressed;
use crate::session::ZendSession;

/// One projection point's placements.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PointPlacements {
    /// The decoded-token position the point was selected at.
    pub start_token: u32,
    pub placed: Vec<Placed>,
    /// The knowledge base's files the point placed, by path, in the order they
    /// sit; a file whose record this member no longer holds is named by its
    /// key.
    pub files: Vec<String>,
}

/// One working piece the point placed turns of.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Placed {
    /// The cap it filled under: `Reading`, `Folder`, `Guide` or `Journal`.
    pub kind: String,
    /// The file or folder it reads; absent for a piece that reads none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub tokens: usize,
}

/// The placements of every point of `turns`, a turn's points in order, each
/// file named through `paths`.
pub fn placements_of<'a>(
    turns: impl IntoIterator<Item = impl IntoIterator<Item = &'a ProjectionEvent>>,
    paths: &HashMap<FileKey, String>,
) -> Vec<Vec<PointPlacements>> {
    turns
        .into_iter()
        .map(|points| {
            points
                .into_iter()
                .map(|event| PointPlacements {
                    start_token: event.start_token,
                    placed: event
                        .pieces
                        .iter()
                        .filter(|p| p.placed > 0)
                        .map(|p| Placed {
                            kind: p.kind.clone(),
                            path: p.standing.path.clone(),
                            tokens: p.placed_tokens,
                        })
                        .collect(),
                    files: event
                        .files
                        .iter()
                        .map(|f| paths.get(f).cloned().unwrap_or_else(|| f.hex()))
                        .collect(),
                })
                .collect()
        })
        .collect()
}

pub async fn get(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let turns = session.conversation_projections(&id);
    let paths = session.knowledge_base_paths();
    let placements = placements_of(
        turns.iter().map(|points| points.iter().map(|p| &p.event)),
        &paths,
    );
    compressed::json(&placements, &headers)
}

#[cfg(test)]
mod tests {
    use candle_conversation::projection::{PieceReport, Standing};
    use serde_json::json;

    use super::*;

    fn piece(kind: &str, path: Option<&str>, placed: usize, tokens: usize) -> PieceReport {
        PieceReport {
            timeline: 1,
            kind: kind.into(),
            best: 0.0,
            best_left: 0.0,
            scored: 0,
            turns: 2,
            placed,
            placed_tokens: tokens,
            standing: Standing {
                path: path.map(str::to_string),
                ..Standing::default()
            },
        }
    }

    /// **A point lists the pieces it placed turns of**, by kind and path, at
    /// its decoded-token position; a piece the fill weighed and placed nothing
    /// of is left out.
    #[test]
    fn a_point_names_what_it_placed() {
        let opening = ProjectionEvent {
            start_token: 0,
            pieces: vec![
                piece("Reading", Some("candle/zend/src/a.rs"), 2, 900),
                piece("Reading", Some("candle/zend/src/b.rs"), 0, 0),
                piece("Journal", None, 1, 40),
            ],
            ..ProjectionEvent::default()
        };
        let later = ProjectionEvent {
            start_token: 128,
            files: vec![FileKey(0x0a), FileKey(0x0b)],
            ..ProjectionEvent::default()
        };
        let paths = HashMap::from([(FileKey(0x0a), "candle/clippy.toml".to_string())]);
        let out = placements_of([vec![&opening, &later]], &paths);
        assert_eq!(
            serde_json::to_value(&out).unwrap(),
            json!([[
                {"start_token": 0, "placed": [
                    {"kind": "Reading", "path": "candle/zend/src/a.rs", "tokens": 900},
                    {"kind": "Journal", "tokens": 40},
                ], "files": []},
                {"start_token": 128, "placed": [],
                 "files": ["candle/clippy.toml", "000000000000000b"]},
            ]])
        );
    }
}
