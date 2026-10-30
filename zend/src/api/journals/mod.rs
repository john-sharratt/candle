//! The memory's journals over HTTP (`docs/memory_backbone.md` §6.7): what
//! the `journal_*` tools do, for a person editing their journals by hand or a
//! program writing many lines at once. Every request is the signed-in
//! caller's, as the gateway names them — listed in the estate's roles, as
//! `/v1/mcp` requires — and reads and writes exactly what their AIs may
//! (§7.2): their own folder and their groups'.
//!
//! | Module | |
//! |---|---|
//! | [`read`] | `GET /v1/journals`, `/v1/journals/tree`, `/v1/journals/grep` |
//! | [`write`] | `POST /v1/journals`: a batch of operations, checked whole, then committed one by one |
//! | [`ops`] | the operations a batch holds, read into the journals crate's |
//! | [`view`] | journals, lines and hits as JSON |
//! | [`import`] | the same batches applied by hand onto a checkout, by whoever holds push access |
//!
//! Every refusal is `{error, message}` under the status that says what to do
//! about it: 400 malformed, 401 signed out, 403 not the caller's to change or
//! an account in no group, 404 no such journal or line, 409 a conflict or a
//! broken file, 422 the rules refuse it, 503 the memory not served here yet —
//! asked again, it may be answered by another member.

pub mod import;
pub mod ops;
pub mod read;
pub mod view;
pub mod write;

use std::fmt::Display;

use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use journals::error::Error;
use journals::scope::Client;
use journals::store::StoreError;
use serde_json::{json, Value};

use crate::mcp::caller::Caller;
use crate::mcp::memory::WriteError;
use crate::mcp::tools::mem::{lasting, Mem};
use crate::session::ZendSession;

/// What a route answers: a status and a body.
pub type Answer = (StatusCode, Value);

/// How long a client waits before asking again what was unavailable.
const RETRY_SECS: &str = "5";

/// A refusal's body.
pub fn refusal(status: StatusCode, error: &str, message: &str) -> Answer {
    (status, json!({ "error": error, "message": message }))
}

/// `answer` as a response; an unavailable one says when to ask again.
pub fn respond((status, body): Answer) -> Response {
    let mut r = (status, Json(body)).into_response();
    if status == StatusCode::SERVICE_UNAVAILABLE {
        r.headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from_static(RETRY_SECS));
    }
    r
}

/// A request that did not parse, in the shape every refusal takes.
pub fn malformed(why: impl Display) -> Response {
    respond(refusal(
        StatusCode::BAD_REQUEST,
        "invalid",
        &why.to_string(),
    ))
}

/// The status and code the journals' refusal `e` answers with.
pub fn status_of_journal(e: &Error) -> (StatusCode, &'static str) {
    match e {
        Error::Invalid(_) => (StatusCode::BAD_REQUEST, "invalid"),
        Error::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
        Error::Forbidden(_) => (StatusCode::FORBIDDEN, "forbidden"),
        Error::Refused(_) => (StatusCode::UNPROCESSABLE_ENTITY, "refused"),
        Error::Conflict(_) => (StatusCode::CONFLICT, "conflict"),
        Error::Broken(_) => (StatusCode::CONFLICT, "broken"),
    }
}

/// The status and code a refused write answers with.
pub fn status_of(e: &WriteError) -> (StatusCode, &'static str) {
    match e {
        WriteError::Unavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "unavailable"),
        WriteError::Store(StoreError::Journal(j)) => status_of_journal(j),
        WriteError::Store(_) => (StatusCode::INTERNAL_SERVER_ERROR, "store"),
    }
}

/// Why the memory has no person for a caller: an account in no group, which
/// stays true however often it is asked, or the memory not served here yet.
pub fn no_person(why: String) -> Answer {
    if lasting(&why) {
        refusal(StatusCode::FORBIDDEN, "stranger", &why)
    } else {
        refusal(StatusCode::SERVICE_UNAVAILABLE, "unavailable", &why)
    }
}

/// The caller as a person, through `client`; or why not.
pub fn person(session: &ZendSession, headers: &HeaderMap, client: Client) -> Result<Mem, Answer> {
    let caller = Caller::of(headers, session.roles()).ok_or_else(|| {
        refusal(
            StatusCode::UNAUTHORIZED,
            "signed_out",
            "sign in at the gateway: the journals are a person's",
        )
    })?;
    Mem::of(session, &caller, client).map_err(no_person)
}

/// `f` on the blocking pool: a caller's first call binds their account, and
/// every write commits, so each request may meet git.
pub async fn blocking(f: impl FnOnce() -> Answer + Send + 'static) -> Response {
    respond(tokio::task::spawn_blocking(f).await.unwrap_or_else(|e| {
        refusal(
            StatusCode::INTERNAL_SERVER_ERROR,
            "panicked",
            &e.to_string(),
        )
    }))
}

#[cfg(test)]
mod tests {
    use journals::error::Problem;

    use super::*;
    use crate::mcp::memory::who::STRANGER;
    use crate::mcp::memory::NO_MEMORY;

    /// **Each refusal answers its own status**, so a program tells a
    /// conflict it may retry from a refusal it must not.
    #[test]
    fn each_refusal_answers_its_own_status() {
        let s = |e: Error| status_of(&WriteError::Store(StoreError::Journal(e)));
        assert_eq!(s(Error::invalid("x")), (StatusCode::BAD_REQUEST, "invalid"));
        assert_eq!(
            s(Error::not_found("x")),
            (StatusCode::NOT_FOUND, "not_found")
        );
        assert_eq!(
            s(Error::Forbidden("x".into())),
            (StatusCode::FORBIDDEN, "forbidden")
        );
        assert_eq!(
            s(Error::refused("x")),
            (StatusCode::UNPROCESSABLE_ENTITY, "refused")
        );
        assert_eq!(s(Error::conflict("x")), (StatusCode::CONFLICT, "conflict"));
        assert_eq!(
            s(Error::Broken(Problem::new("f", None, "bad"))),
            (StatusCode::CONFLICT, "broken")
        );
        assert_eq!(
            status_of(&WriteError::Unavailable("x".into())),
            (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
        );
        assert_eq!(
            status_of(&WriteError::Store(StoreError::Repository("x".into()))),
            (StatusCode::INTERNAL_SERVER_ERROR, "store")
        );
    }

    /// **An account in no group is told so for good**; a memory not served
    /// here yet is asked again.
    #[test]
    fn a_stranger_is_forbidden_and_a_memory_not_served_is_unavailable() {
        assert_eq!(
            no_person(STRANGER.into()),
            (
                StatusCode::FORBIDDEN,
                json!({"error": "stranger", "message": STRANGER})
            )
        );
        assert_eq!(
            no_person(NO_MEMORY.into()),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"error": "unavailable", "message": NO_MEMORY})
            )
        );
    }

    #[test]
    fn an_unavailable_answer_says_when_to_ask_again() {
        let r = respond(refusal(StatusCode::SERVICE_UNAVAILABLE, "unavailable", "x"));
        assert_eq!(r.headers()[RETRY_AFTER], RETRY_SECS);
        let r = respond(refusal(StatusCode::NOT_FOUND, "not_found", "x"));
        assert!(r.headers().get(RETRY_AFTER).is_none());
    }
}
