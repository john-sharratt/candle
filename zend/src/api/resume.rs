//! `POST /v1/conversations/{id}/resume` — resume a conversation's request that
//! a stop cut off (`session::ZendSession::resume_request`): by its resume
//! entries, or, when those were retracted, asked again from its opening
//! message. Answers at once; the request runs on, followed as a live turn.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;

use crate::session::{ResumeAsked, ZendSession};

/// The status and line for what the resume found.
fn answer(asked: ResumeAsked) -> (StatusCode, &'static str) {
    match asked {
        ResumeAsked::Resuming => (StatusCode::ACCEPTED, "resuming its request"),
        ResumeAsked::Live => (StatusCode::CONFLICT, "a turn of it is running now"),
        ResumeAsked::Finished => (StatusCode::CONFLICT, "its request ended: nothing to resume"),
        ResumeAsked::NotLoaded => (
            StatusCode::SERVICE_UNAVAILABLE,
            "the model is not loaded yet",
        ),
    }
}

pub async fn resume(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
) -> (StatusCode, &'static str) {
    answer(session.resume_request(&id))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Only a cut-off request is started**; one running, ended, or on a
    /// member not yet loaded is refused, saying which.
    #[test]
    fn only_a_cut_off_request_is_started() {
        assert_eq!(answer(ResumeAsked::Resuming).0, StatusCode::ACCEPTED);
        assert_eq!(answer(ResumeAsked::Live).0, StatusCode::CONFLICT);
        assert_eq!(answer(ResumeAsked::Finished).0, StatusCode::CONFLICT);
        assert_eq!(
            answer(ResumeAsked::NotLoaded).0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
