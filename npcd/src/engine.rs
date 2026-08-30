//! The surface an inference engine would answer, wired and honest about being
//! empty.
//!
//! # Why these are routes at all
//!
//! Everything here needs something to have *run*: a substrate with turns in it,
//! a projection composed for a tick, a monitor that scored an overlap, a model
//! that generated a portrait. This daemon runs none of that yet.
//!
//! They used to fall through to the console's fixture, which answered every one
//! of them with invented data — for any character id, including ones that do
//! not exist. That is the worst shape a fixture can take: not obviously fake,
//! in the place the real thing belongs. A reader could not tell a substrate
//! with nothing in it from one they were being shown a story about.
//!
//! So the routes are here, they check the things that *can* be checked — the
//! character exists, the caller owns it, the layer is one the schema declares —
//! and then they say what is true:
//!
//! - **Empty, where empty is the measurement.** A character that has never run
//!   has no turns in any layer. `[]` is the honest answer and the console
//!   renders it as "nothing in this layer yet".
//! - **Absent, where nothing has measured.** A projection, a monitor band, a
//!   resident percentage. `null`, never `0` — a zero is a measurement, and
//!   reporting one nothing took is the fabrication this module exists to end.
//! - **`503 no_engine`, where the request asks for work.** Probing retrieval,
//!   opening an interaction, generating a portrait. A refusal that names what
//!   is missing, rather than a job id that will never complete.
//!
//! # What changes when there is an engine
//!
//! These handlers, and nothing else. The routes, their roles, their shapes and
//! the console above them are already what they will be — which is the point of
//! wiring them now rather than leaving them to the fixture.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use serde_json::{json, Value};
use web::auth::Role;

use crate::api::{err, owner_of, Authored};
use crate::guard::Api;
use crate::projection;

/// The one refusal this module makes, worded once.
///
/// `503`, not `501`: the route exists and is correct, and the thing it needs is
/// expected to arrive. A `501` would say the daemon does not implement it,
/// which is the wrong thing to tell somebody whose console is otherwise working.
pub fn no_engine(what: &str) -> Response {
    err(
        StatusCode::SERVICE_UNAVAILABLE,
        "no_engine",
        &format!("{what} needs an inference engine, and this daemon is not running one"),
    )
}

pub fn api(state: Arc<Authored>) -> Api<Arc<Authored>> {
    Api::new(state.roles.clone())
        // ── the substrate, as it actually is ────────────────────────────────
        .route("/v1/npc/:nid/substrate", Role::User, get(substrate))
        .route(
            "/v1/npc/:nid/substrate/layer/:layer",
            Role::User,
            get(layer),
        )
        .route(
            "/v1/npc/:nid/substrate/turn/:layer/:turn",
            Role::User,
            get(turn),
        )
        .route("/v1/npc/:nid/memory", Role::User, get(memory))
        // ── instruments ─────────────────────────────────────────────────────
        .route("/v1/npc/:nid/projection", Role::User, get(projection_now))
        .route(
            "/v1/npc/:nid/projection/:tick",
            Role::User,
            get(projection_at),
        )
        .route("/v1/npc/:nid/monitor", Role::User, get(monitor))
        .route("/v1/npc/:nid/project", Role::User, post(probe))
        .route("/v1/npc/:nid/perceive", Role::User, post(perceive))
        // ── interactions ────────────────────────────────────────────────────
        .route(
            "/v1/npc/:nid/interaction",
            Role::User,
            get(list_interactions).post(open_interaction),
        )
        .route(
            "/v1/interaction/:ix",
            Role::User,
            get(interaction).delete(end_interaction),
        )
        .route("/v1/interaction/:ix/inject", Role::User, post(inject))
        .route("/v1/interaction/:ix/stream", Role::User, get(stream))
        // ── the act vocabulary ──────────────────────────────────────────────
        .route("/v1/tools", Role::User, get(tools))
        .route("/v1/tools/calibrate", Role::Admin, post(calibrate))
        .route("/v1/commands", Role::User, get(commands))
        // ── generation ──────────────────────────────────────────────────────
        .route(
            "/v1/generate/description",
            Role::User,
            post(gen_description),
        )
        .route("/v1/generate/attributes", Role::User, post(gen_attributes))
        .route("/v1/image/generate", Role::User, post(gen_image))
        .route("/v1/image/models", Role::User, get(image_models))
        .route("/v1/image/queue", Role::User, get(image_queue))
        // ── the push stream ─────────────────────────────────────────────────
        .route("/ws/events", Role::User, get(events))
}

/// Live character state — ticks, monitor bands, inbox depth.
///
/// A real socket that connects, holds, and says nothing, because nothing is
/// emitting. The console's roster subscribes to this to light its state dots,
/// and a socket that accepts and stays quiet is exactly right: the dots stay as
/// the listing drew them.
///
/// Not a 503, unlike the routes above. A refusal here would put the console
/// into its reconnect loop — backing off, retrying, reporting a fault — over a
/// daemon that is working perfectly and simply has nothing to say yet.
async fn events(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(|mut socket| async move {
        // Held open until the client goes away. Reading is what notices that:
        // a browser closing a tab sends a close frame, and without this the
        // task would linger until the process ended.
        while let Some(Ok(msg)) = socket.recv().await {
            if matches!(msg, Message::Close(_)) {
                return;
            }
        }
    })
}

/// Confirm the caller owns this character, and give back its id.
///
/// Every route here is about one character, so every one of them 404s for a
/// character that does not exist — which is the difference the fixture could
/// not make, since it answered for any id at all.
async fn owned(s: &Arc<Authored>, headers: &HeaderMap, nid: &str) -> Result<u64, Response> {
    let (_, owner) = owner_of(s, headers).await.map_err(|r| *r)?;
    let Ok(npc_id) = nid.parse::<u64>() else {
        return Err(err(
            StatusCode::NOT_FOUND,
            "npc_not_found",
            "no such character",
        ));
    };
    if s.npcs.read().await.visible_to(npc_id, &owner).is_none() {
        return Err(err(
            StatusCode::NOT_FOUND,
            "npc_not_found",
            "no such character",
        ));
    }
    Ok(npc_id)
}

/// The layer occupancy: every layer the schema declares, and what is in it.
///
/// Nothing is, yet. The turn and token counts are real zeros — a character that
/// has never run genuinely holds no turns — while `resident` is a measurement
/// of paging that nothing has taken, so it is absent.
async fn substrate(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    let layers = projection::layers(&s.mind).unwrap_or_default();
    Json(json!({
        "layers": layers.iter().map(|l| json!({
            "layer": l.get("name").cloned().unwrap_or(Value::Null),
            "window": l.get("window").cloned().unwrap_or(Value::Null),
            "turns": 0,
            "tokens": 0,
            // How much of this layer is resident in VRAM. A paging figure, and
            // nothing has paged anything.
            "resident": Value::Null,
        })).collect::<Vec<_>>(),
        "engine_connected": false,
    }))
    .into_response()
}

/// One layer's turns. None, and the layer has to be one that exists.
async fn layer(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, name)): Path<(String, String)>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    let layers = projection::layers(&s.mind).unwrap_or_default();
    let known = layers
        .iter()
        .any(|l| l.get("name").and_then(Value::as_str) == Some(name.as_str()));
    if !known {
        // Checked, because it can be: a typo in a layer name should be a 404
        // rather than an empty list that looks like an empty layer.
        return err(
            StatusCode::NOT_FOUND,
            "no_such_layer",
            &format!("`{name}` is not a layer this mind declares"),
        );
    }
    Json(json!({ "layer": name, "items": [], "engine_connected": false })).into_response()
}

/// One turn's stored form. There are no turns, so there is no turn.
async fn turn(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, _layer, _turn)): Path<(String, String, String)>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    err(
        StatusCode::NOT_FOUND,
        "turn_not_found",
        "this character has no turns — nothing has run",
    )
}

/// What the character remembers having lived. Nothing has happened to it.
async fn memory(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    Json(json!({ "items": [], "next_cursor": Value::Null, "engine_connected": false }))
        .into_response()
}

async fn projection_now(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    projection_absent(&s, &headers, &nid).await
}

async fn projection_at(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((nid, _tick)): Path<(String, String)>,
) -> Response {
    projection_absent(&s, &headers, &nid).await
}

/// A projection is a composition made for one tick. None has been made.
///
/// `404` rather than an empty budget: an empty projection would say the gather
/// ran and found nothing, which is a different and much more alarming claim
/// than "nothing has run".
async fn projection_absent(s: &Arc<Authored>, headers: &HeaderMap, nid: &str) -> Response {
    if let Err(r) = owned(s, headers, nid).await {
        return r;
    }
    err(
        StatusCode::NOT_FOUND,
        "no_projection",
        "no projection has been composed for this character — nothing has run",
    )
}

/// The metacognition monitor's band and overlap trace.
async fn monitor(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    // Absent, not `healthy`. A band is a verdict on a character's attention,
    // and this one has had none — the console renders `null` as "not measured".
    Json(json!({
        "band": Value::Null,
        "overlap": Value::Null,
        "engine_connected": false,
    }))
    .into_response()
}

async fn probe(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    no_engine("probing retrieval")
}

async fn perceive(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    no_engine("delivering an event to a character")
}

/// Interactions this character is in. None, because opening one needs an engine.
async fn list_interactions(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    Json(json!({ "interactions": [], "engine_connected": false })).into_response()
}

async fn open_interaction(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return r;
    }
    // Opening one forks the character's substrate and starts a decode loop.
    no_engine("opening an interaction")
}

/// There are no interactions, so no id names one.
async fn interaction(Path(ix): Path<String>) -> Response {
    err(StatusCode::NOT_FOUND, "interaction_not_found", &ix)
}

async fn end_interaction(Path(ix): Path<String>) -> Response {
    err(StatusCode::NOT_FOUND, "interaction_not_found", &ix)
}

async fn inject(Path(_ix): Path<String>) -> Response {
    no_engine("speaking to a character")
}

async fn stream(Path(_ix): Path<String>) -> Response {
    no_engine("streaming an interaction")
}

/// The act vocabulary.
///
/// Empty, and `engine_connected: false` beside it so the console can say *why*
/// it is empty. The tools are registered by the engine with the layers each may
/// write, and calibration is a pass it runs; there is no authored catalog in the
/// mind to read one from instead.
async fn tools() -> Response {
    Json(json!({
        "tools": [],
        // Absent, not zero. Zero uncalibrated tools out of zero tools is a
        // green tick over an empty table, which is exactly what the console
        // used to draw when this request failed.
        "uncalibrated": Value::Null,
        "engine_connected": false,
    }))
    .into_response()
}

async fn calibrate() -> Response {
    no_engine("calibrating tools")
}

async fn commands() -> Response {
    Json(json!({ "commands": [], "engine_connected": false })).into_response()
}

async fn gen_description() -> Response {
    no_engine("writing a description")
}

async fn gen_attributes() -> Response {
    no_engine("generating attributes")
}

async fn gen_image() -> Response {
    no_engine("generating an image")
}

/// Image models this daemon could run. It loads none.
async fn image_models() -> Response {
    Json(json!({ "models": [], "engine_connected": false })).into_response()
}

/// The image queue. There is no queue, which is not the same as an empty one —
/// an empty queue implies something that would run it.
async fn image_queue() -> Response {
    Json(json!({
        "depth": Value::Null,
        "position": Value::Null,
        "state": Value::Null,
        "engine_connected": false,
    }))
    .into_response()
}
