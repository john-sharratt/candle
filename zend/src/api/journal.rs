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
//!
//! The journal is read-only from outside: every line in it is one the server
//! cited and checked, so an entry can be forgotten but not edited.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
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

    #[tokio::test]
    async fn an_unloaded_model_is_unavailable() {
        let (status, body) = body(not_loaded()).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"], "not_loaded");
    }
}
