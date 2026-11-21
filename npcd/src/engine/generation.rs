//! Routes the generations the console asks for when building a world (description,
//! name, image and the model list), each answered by its real guest.
//!
//! These routes used to answer 503 or an empty list, so the console fell back to
//! its own placeholder text and an author read that as the generator being
//! unavailable. The image route forwards the caller's headers, because the image
//! guest re-checks a restricted role itself, so a draw is authorised whichever
//! route it came in by. The description route streams a fragment at a time for
//! the create form. The queue route reports that there is no queue, rather than an
//! empty one, because an empty queue would imply a worker that runs it.

use super::*;

/// Real: written by the prose guest, against the world's own setting.
///
/// This was a 503 stub and the console's create page has called it since it was
/// written — the page's fallback text ("generation unavailable — write one
/// yourself") is what an author saw instead. See [`crate::describe`].
pub(super) async fn gen_description(
    state: State<Arc<crate::api::Authored>>,
    body: Json<crate::describe::DescribeBody>,
) -> Response {
    crate::describe::post_describe(state, body).await
}

/// The same generation, streamed a fragment at a time. See
/// [`crate::describe::post_describe_stream`].
pub(super) async fn gen_description_stream(
    state: State<Arc<crate::api::Authored>>,
    body: Json<crate::describe::DescribeBody>,
) -> Response {
    crate::describe::post_describe_stream(state, body).await
}

/// Real: named by the prose guest, against the world's own summary. See
/// [`crate::namegen`].
pub(super) async fn gen_name(
    state: State<Arc<crate::api::Authored>>,
    body: Json<crate::namegen::NameBody>,
) -> Response {
    crate::namegen::post_name(state, body).await
}

pub(super) async fn gen_attributes() -> Response {
    no_engine("generating attributes")
}

/// Real: drawn by the image guest.
///
/// This answered 503 while `/v1/guest/image` served the same request, so the
/// console's own "draw me an image" route was the one thing on the daemon that
/// could not. The create step needs it: it draws a portrait to show *before*
/// there is a character to address `/v1/npc/:nid/portrait/generate` to.
pub(super) async fn gen_image(
    state: State<Arc<crate::api::Authored>>,
    headers: axum::http::HeaderMap,
    body: Json<crate::guest_routes::ImageBody>,
) -> Response {
    // The headers ride along because `post_image` re-checks the caller's role
    // itself for a `restricted` draw — whichever route the request came in by.
    crate::guest_routes::post_image(state, headers, body).await
}

/// Real: the image guest this deployment has configured, if any.
///
/// It used to answer an empty list unconditionally, so the console's create
/// step inferred "no image model is loaded" — true at the time, and a guess
/// rather than a report. See [`crate::portrait::get_models`].
pub(super) async fn image_models(state: State<Arc<crate::api::Authored>>) -> Response {
    crate::portrait::get_models(state).await
}

/// The image queue. There is no queue, which is not the same as an empty one —
/// an empty queue implies something that would run it.
pub(super) async fn image_queue() -> Response {
    Json(json!({
        "depth": Value::Null,
        "position": Value::Null,
        "state": Value::Null,
        "engine_connected": false,
    }))
    .into_response()
}
