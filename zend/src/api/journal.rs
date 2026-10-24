//! A dialogue's working journal (`docs/zend_journal.md` §8).
//!
//! `GET /v1/conversations/{id}/journal` — the entries as the prompt reads them,
//!   the open items, what the journal covers, and its recent drafts with each
//!   one's timings and each claim's verdict and kind.
//! `POST /v1/conversations/{id}/journal/draft` — draft now: `202` with the turns
//!   the draft covers, `409` while one runs, when nothing is new, or when the
//!   conversation is not open on this member.
//! `DELETE /v1/conversations/{id}/journal/{eid}` — forget one entry: the journal
//!   as it now reads, or `404` when no entry by that number is held.
//! `GET /v1/conversations/{id}/journal/{eid}/turns?offset=&limit=` — the turns
//!   behind one entry exactly as the dialogue's `journal_read` pages them (a
//!   folded entry's digest for a folded one): `limit` lines (default 200, at
//!   most 400) from line `offset` (default 1); `404` with the miss's code when
//!   no text is kept for it.
//!
//! The journal is read-only from outside: every line in it is one the server
//! cited and checked, so an entry can be forgotten but not edited.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use journal_core::transcript::{Miss, MAX_LINES};
use serde::Deserialize;
use serde_json::json;

use crate::session::{NoDraft, ZendSession};

fn refused(status: StatusCode, error: &str, message: &str) -> Response {
    (status, Json(json!({ "error": error, "message": message }))).into_response()
}

fn not_loaded() -> Response {
    refused(
        StatusCode::SERVICE_UNAVAILABLE,
        "not_loaded",
        "the model is not loaded yet",
    )
}

pub async fn get(State(session): State<Arc<ZendSession>>, Path(id): Path<String>) -> Response {
    match session.journal(&id) {
        Some(view) => Json(view).into_response(),
        None => not_loaded(),
    }
}

pub async fn draft(State(session): State<Arc<ZendSession>>, Path(id): Path<String>) -> Response {
    match session.journal_draft(&id) {
        Some(Ok(covers)) => (StatusCode::ACCEPTED, Json(covers)).into_response(),
        Some(Err(why)) => no_draft(why),
        None => not_loaded(),
    }
}

fn no_draft(why: NoDraft) -> Response {
    match why {
        NoDraft::NotOpen => refused(
            StatusCode::CONFLICT,
            "not_open",
            "the conversation is not open on this member — a draft forks its live slot, so \
             send it a message first",
        ),
        NoDraft::NotDue | NoDraft::NothingNew => refused(
            StatusCode::CONFLICT,
            "nothing_to_draft",
            "a draft is already running, or nothing has happened since the journal last \
             looked",
        ),
    }
}

pub async fn forget(
    State(session): State<Arc<ZendSession>>,
    Path((id, entry)): Path<(String, u64)>,
) -> Response {
    match session.journal_forget(&id, entry) {
        Some(Ok(Some(view))) => Json(view).into_response(),
        Some(Ok(None)) => refused(
            StatusCode::NOT_FOUND,
            "no_such_entry",
            "the journal holds no entry by that number",
        ),
        Some(Err(e)) => refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "journal_not_recorded",
            &e.to_string(),
        ),
        None => not_loaded(),
    }
}

/// Lines a page holds when the request names no count.
const DEFAULT_LINES: usize = 200;

#[derive(Deserialize)]
pub struct TurnsQuery {
    offset: Option<usize>,
    limit: Option<usize>,
}

pub async fn turns(
    State(session): State<Arc<ZendSession>>,
    Path((id, entry)): Path<(String, u64)>,
    Query(q): Query<TurnsQuery>,
) -> Response {
    let (from, lines) = page_of(&q);
    match session.journal_turns(&id, entry, from, lines) {
        Some(Ok(page)) => Json(page).into_response(),
        Some(Err(miss)) => missed(&miss),
        None => not_loaded(),
    }
}

/// The page a query asks for: its first line, and how many, at most
/// [`MAX_LINES`] as the tool allows.
fn page_of(q: &TurnsQuery) -> (usize, usize) {
    (
        q.offset.unwrap_or(1).max(1),
        q.limit.unwrap_or(DEFAULT_LINES).clamp(1, MAX_LINES),
    )
}

fn missed(miss: &Miss) -> Response {
    refused(StatusCode::NOT_FOUND, miss.code(), &miss.detail())
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;

    use super::*;

    async fn body(response: Response) -> (StatusCode, serde_json::Value) {
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn a_draft_that_cannot_fork_says_so_and_conflicts() {
        let (status, body) = body(no_draft(NoDraft::NotOpen)).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"], "not_open");
    }

    #[tokio::test]
    async fn a_draft_with_nothing_to_cover_conflicts() {
        for why in [NoDraft::NotDue, NoDraft::NothingNew] {
            let (status, body) = body(no_draft(why)).await;
            assert_eq!(status, StatusCode::CONFLICT);
            assert_eq!(body["error"], "nothing_to_draft");
        }
    }

    #[test]
    fn a_page_defaults_to_the_first_two_hundred_lines_and_is_bounded_as_the_tool_is() {
        let q = |offset, limit| TurnsQuery { offset, limit };
        assert_eq!(page_of(&q(None, None)), (1, 200));
        assert_eq!(page_of(&q(Some(0), Some(0))), (1, 1));
        assert_eq!(page_of(&q(Some(41), Some(9_999))), (41, MAX_LINES));
    }

    #[tokio::test]
    async fn an_entry_with_no_text_is_not_found_under_the_tools_code() {
        let (status, body) = body(missed(&Miss::Entry(7))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"], "no_entry_text");
        assert!(body["message"].as_str().unwrap().contains("entry 7"));
    }

    #[tokio::test]
    async fn an_unloaded_model_is_unavailable() {
        let (status, body) = body(not_loaded()).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"], "not_loaded");
    }
}
