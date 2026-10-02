//! The route envelope: one mapping from an act's [`Outcome`] to an HTTP
//! [`Response`], reused by every migrated device route.
//!
//! A device route (§9) is an `enact.rs`/`work.rs` handler with an HTTP envelope
//! around it: it synthesises the [`Act`](crate::engine::act::Act) the real
//! dispatch already knows how to run, runs it through
//! [`crate::engine::body::perform`], and hands the world's verdict back as a
//! response. The verdict is an [`Outcome`], and turning that into a status and a
//! JSON body is the same job for every route — so it lives here, once, rather
//! than in each namespace's handlers.
//!
//! The mapping, whole:
//!
//! - [`Outcome::Did`] / [`Outcome::Departed`] — the act landed. `200` with
//!   `{"ok": true, "detail": <the line the character reads>}`. The two are one
//!   case here: a device route does not narrate a parting, so the distinction
//!   `record_act` draws for the feed does not reach the wire.
//! - [`Outcome::Refused`] — the world would not have it, and said why. `409`
//!   with the estate error shape `{"error": "refused", "detail": <the why>}`
//!   ([`err`]), so a wrong `invoke` comes back as an ordinary REST error the
//!   model corrects against (§12), the refusal's own prose in `detail`.
//! - [`Outcome::NotOfTheBody`] — `500`. A route synthesises a body act it knows
//!   the dispatch performs, so this is never a caller's mistake to report as a
//!   `4xx`; it is the route having built an act the body does not perform, which
//!   is a bug in the route, not in the request.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::api::err;
use crate::engine::body::Outcome;

/// Map the world's verdict on a synthesised act to the route's response.
///
/// See the module docs for the whole mapping. The `detail` on a success is the
/// character-facing line the act came back with — the same words a character
/// reads in its `<tool_response>` — so the in-fiction fast path and an external
/// client read one answer.
pub fn enact_response(outcome: Outcome) -> Response {
    match outcome {
        Outcome::Did(line) | Outcome::Departed(line) => {
            (StatusCode::OK, Json(json!({ "ok": true, "detail": line }))).into_response()
        }
        Outcome::Refused(line) => err(StatusCode::CONFLICT, "refused", &line),
        // A route only ever synthesises an act the dispatch performs, so a
        // non-body act here is the route's own bug, not a bad request.
        Outcome::NotOfTheBody => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "not_of_the_body",
            "this route synthesised an act the body does not perform",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a response into its status and parsed JSON body.
    async fn parts(response: Response) -> (StatusCode, serde_json::Value) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// A landed act is `200 {ok:true, detail:<line>}`, carrying the character's
    /// own line unchanged.
    #[tokio::test]
    async fn a_did_is_two_hundred_ok_with_its_line() {
        let (status, body) = parts(enact_response(Outcome::Did(
            "The lift is on its way.".into(),
        )))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            json!({ "ok": true, "detail": "The lift is on its way." })
        );
    }

    /// A departing act maps the same as a plain landed one — the parting is a
    /// narration concern that never reaches the wire.
    #[tokio::test]
    async fn a_departed_maps_like_a_did() {
        let (status, body) = parts(enact_response(Outcome::Departed(
            "You call after them.".into(),
        )))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            json!({ "ok": true, "detail": "You call after them." })
        );
    }

    /// A refusal is `409` with the estate `{error, detail}` shape, the `error`
    /// naming the class and the `detail` carrying the world's own why.
    #[tokio::test]
    async fn a_refused_is_four_oh_nine_in_the_estate_error_shape() {
        let (status, body) = parts(enact_response(Outcome::Refused(
            "You are not at the lift. Make your way to it first.".into(),
        )))
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(
            body,
            json!({
                "error": "refused",
                "detail": "You are not at the lift. Make your way to it first."
            })
        );
    }

    /// A non-body act is a `500` — the route built an act the body does not
    /// perform, which is the route's bug, not the request's.
    #[tokio::test]
    async fn a_not_of_the_body_is_a_five_hundred() {
        let (status, body) = parts(enact_response(Outcome::NotOfTheBody)).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"], "not_of_the_body");
    }
}
