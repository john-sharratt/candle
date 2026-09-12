//! The engine: the cast's loop, the act vocabulary, and the routes above them.
//!
//! # What is live
//!
//! The whole perception half and the loop that drives it. Characters are woken
//! at startup, tick on their own salience-driven heartbeats, drain their inboxes
//! as prose, decode through [`mind`], and roll their conversation over at the day
//! boundary. [`pulse`] is the instrument for watching all of it; [`slash`] is how
//! an operator puts an event into a character's inbox by hand.
//!
//! Submodules, roughly in the order a character meets them:
//!
//! | Module | What it owns |
//! |---|---|
//! | [`loading`] | the startup phases the console's loading screen reads |
//! | [`runtime`] | standing the engine up, and the thread that keeps it running |
//! | [`event`] | what arrives in an inbox, and the prose it becomes |
//! | [`tick`] | the scheduler: every character's loop, and what runs when |
//! | [`window`] | the bounded verbatim tail carried into the next decode |
//! | [`persona`] | an authored record rendered as the person the model reads |
//! | [`prompt`] | the lens: identity, beliefs, vocabulary, the call format |
//! | [`tools`] | the act vocabulary, with its calibration examples |
//! | [`act`] | reading acts back out of what the character said |
//! | [`mind`] | the per-character conversation and the decode itself |
//! | [`sleep`] | the day boundary: tombstone yesterday, open today |
//! | [`watcher`] | the mind directory, reloaded without a restart |
//! | [`slash`] | `/` commands — an operator speaking to the loop |
//! | [`pulse`] | the routes the Pulse view reads |
//!
//! # What is still absent, and says so
//!
//! Some routes here need machinery that does not exist yet: a projection
//! composed for a tick, a monitor that scored an overlap, a model that generated
//! a portrait. Those answer honestly rather than plausibly, which is the rule
//! this module was written to enforce:
//!
//! - **Empty, where empty is the measurement.** A character that has never run
//!   has no turns in a layer. `[]` is the honest answer.
//! - **Absent, where nothing has measured.** `null`, never `0` — a zero is a
//!   measurement, and reporting one nothing took is a fabrication.
//! - **`503 no_engine`, where the request asks for work nothing can do yet.**
//!   A refusal that names what is missing, rather than a job id that will never
//!   complete.
//!
//! They used to fall through to the console's fixture, which answered every one
//! of them with invented data for any character id, including ones that did not
//! exist — not obviously fake, and in the place the real thing belongs.

pub mod act;
pub mod acts;
pub mod authoring;
pub mod bench;
pub mod body;
pub mod cooldown;
pub mod driver;
pub mod enact;
pub mod environment;
pub mod event;
pub mod identity;
pub mod ingest;
pub mod interaction;
pub mod life;
pub mod loading;
pub mod mind;
pub mod perceived;
pub mod persona;
pub mod prompt;
pub mod pulse;
pub mod reach;
pub mod reflect;
pub mod retention;
pub mod rooms;
pub mod runtime;
pub mod schema;
pub mod simulate;
pub mod slash;
pub mod sleep;
pub mod station;
pub mod stir;
pub mod tick;
pub mod tools;
pub mod watcher;
pub mod whereabouts;
pub mod window;
pub mod work;

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use serde_json::{json, Value};
use web::auth::Role;

use crate::api::{err, owner_of, Authored};
use crate::engine::interaction::Interlocutor;
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
        // ── the scenario harness ────────────────────────────────────────────
        //
        // Put a situation to a character that does not exist and report
        // everything about what came back. Admin, because it spends the one
        // card's time and reports the daemon's own prompt.
        .route("/v1/simulate", Role::Admin, post(simulate::run))
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
        // ── messaging a character ───────────────────────────────────────────
        //
        // **Real, and the same handset the characters use.** Not a private
        // channel between a console and a mind: a person messaging a character
        // is one more party on a thread, and the character answers with the
        // same `message` act it would use to answer anybody. That is what makes
        // the reply worth having — it is the character speaking to you, from
        // inside the world, rather than a chat window bolted to the side of it.
        .route(
            "/v1/npc/:nid/message",
            Role::User,
            get(get_messages).post(post_message),
        )
        // ── the world's open channel ────────────────────────────────────────
        //
        // The standing group every character joins on arrival — see
        // `sim::phone::CHANNEL`. Registered twice because the two methods need
        // different roles, which is what one line each is for: reading it is a
        // `User`'s, and speaking on it is a `Creator`'s, because it reaches
        // every character in a world at once and ownership cannot express that.
        // See [`post_channel`].
        .route("/v1/world/:wid/channel", Role::User, get(get_channel))
        .route("/v1/world/:wid/channel", Role::Creator, post(post_channel))
        // Words left on something in a world, for whoever comes to it — the
        // world's half of `post_notice`. Creator for the same reason the
        // channel's write side is: it writes into a world rather than into a
        // character somebody owns.
        .route("/v1/world/:wid/posting", Role::Creator, post(post_posting))
        // ── the act vocabulary ──────────────────────────────────────────────
        //
        // Real: the catalog and the `/` command list are compiled in, and both
        // are served rather than duplicated in the console so the two cannot
        // drift. Calibration still needs the engine.
        .route("/v1/tools", Role::User, get(pulse::tools))
        .route("/v1/tools/calibrate", Role::Admin, post(calibrate))
        .route("/v1/commands", Role::User, get(pulse::commands))
        // ── Pulse: the cast's loop, as an instrument ────────────────────────
        .route("/v1/pulse", Role::User, get(pulse::feed))
        .route("/v1/pulse/census", Role::User, get(pulse::census))
        // Where everybody is standing. The one question the feed and the window
        // between them cannot answer — see [`pulse::world`].
        //
        // Under `/v1/pulse`, not `/v1/world`: that one is the authored world
        // registry, which is a different thing entirely — the documents an
        // author wrote, not the simulation running from them.
        .route("/v1/pulse/world", Role::User, get(pulse::world))
        .route("/v1/npc/:nid/pulse", Role::User, post(pulse::inject))
        .route("/v1/npc/:nid/window", Role::User, get(pulse::window))
        // Admin: it reaches characters the caller does not own, which every
        // other route on this daemon refuses to do.
        .route("/v1/pulse/broadcast", Role::Admin, post(pulse::broadcast))
        // ── generation ──────────────────────────────────────────────────────
        .route(
            "/v1/generate/description",
            Role::User,
            post(gen_description),
        )
        // The same generation, arriving a token at a time. Both, because they
        // are different operations to a caller — see [`describe::post_describe_stream`].
        .route(
            "/v1/generate/description/stream",
            Role::User,
            post(gen_description_stream),
        )
        // Named before described: the create form fills the name field the
        // moment it opens, and the description is then written about that
        // person rather than inventing a second one.
        .route("/v1/generate/name", Role::User, post(gen_name))
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
///
/// `Box`ed on the error side, matching [`owner_of`]. An axum `Response` is a
/// large value and a `Result` is as big as its widest arm, so an unboxed one
/// makes every success on this path carry the refusal's footprint.
async fn owned(s: &Arc<Authored>, headers: &HeaderMap, nid: &str) -> Result<u64, Box<Response>> {
    let (_, owner) = owner_of(s, headers).await?;
    let not_found = || {
        Box::new(err(
            StatusCode::NOT_FOUND,
            "npc_not_found",
            "no such character",
        ))
    };
    let Ok(npc_id) = nid.parse::<u64>() else {
        return Err(not_found());
    };
    if s.npcs.read().await.visible_to(npc_id, &owner).is_none() {
        return Err(not_found());
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
        return *r;
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
        return *r;
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
        return *r;
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
        return *r;
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
        return *r;
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
        return *r;
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
        return *r;
    }
    no_engine("probing retrieval")
}

async fn perceive(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    if let Err(r) = owned(&s, &headers, &nid).await {
        return *r;
    }
    no_engine("delivering an event to a character")
}

/// Who this caller is, to a character.
fn as_interlocutor(
    id: &web::auth::session::Identity,
    handle: &str,
    roles: &web::auth::Roles,
) -> Interlocutor {
    Interlocutor {
        kind: "operator".into(),
        id: handle.to_string(),
        display: speaking_as(id, handle, roles),
    }
}

/// Interactions this character is in.
async fn list_interactions(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return Json(json!({ "interactions": [], "engine_connected": false })).into_response();
    };
    // The wall clock: whether a session has gone quiet is a fact about a person
    // at a console, not about the world's own time. See `Interaction::last_ms`.
    let now = crate::api::now_ms();
    let live: Vec<Value> = rt
        .interactions
        .for_npc(npc_id, now)
        .iter()
        .map(|ix| ix.wire(now))
        .collect();
    Json(json!({ "interactions": live, "engine_connected": true })).into_response()
}

/// Open one, or continue the one already open with this person in this mode.
async fn open_interaction(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("opening an interaction");
    };
    // Physical by default: standing in the room together is the ordinary way to
    // be present to a character, and the one an operator who did not say means.
    let wanted = body
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("physical");
    let Some(mode) = crate::engine::tools::Mode::parse(wanted) else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_mode",
            "a mode is physical or instant_message",
        );
    };
    // A character with no body cannot be stood next to. The messaging modes
    // reach somebody who is nowhere near, which is the whole point of them, so
    // only the physical one needs a place.
    if mode == crate::engine::tools::Mode::Physical && rt.body_of(npc_id).is_none() {
        return err(
            StatusCode::CONFLICT,
            "not_in_a_world",
            "that character has no body to stand beside — open a messaging mode instead",
        );
    }
    // Two clocks, and they are not interchangeable: going quiet is a fact about
    // a person at a console, and the world's own instant is only the label the
    // session carries. See `Interaction::last_ms`.
    let world_ms = s.world_ms(npc_id).await;
    let now = crate::api::now_ms();
    let who = as_interlocutor(&id, &handle, &s.roles);
    let world = rt
        .body_of(npc_id)
        .map(|(hosted, _)| hosted.id().to_string());
    let ix = rt
        .interactions
        .open(npc_id, mode, who.clone(), world, now, world_ms);

    // **Physical means physical.** The person walks into the room the character
    // is standing in, so it appears in `Within::company` — which is what turns
    // `tell`, `ask`, `give`, `touch` and `gesture` back on. Without a body it
    // heard you and could not answer you, and went on waiting for somebody to
    // arrive while you were talking to it.
    let standing = match mode == crate::engine::tools::Mode::Physical {
        true => rt.enter_world_beside(npc_id, &ix.body, &who.display),
        false => None,
    };
    let mut wire = ix.wire(now);
    if let Some(at) = standing {
        wire["you_are_at"] = json!(at.to_string());
    }
    Json(wire).into_response()
}

async fn interaction(State(s): State<Arc<Authored>>, Path(ix): Path<String>) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    let now = crate::api::now_ms();
    match rt.interactions.get(&ix, now) {
        Some(found) => Json(found.wire(now)).into_response(),
        // A session that has gone quiet reads as gone rather than as ended:
        // there is nothing left to look at either way, and the console's own
        // "no such interaction" is the honest thing to show.
        None => err(StatusCode::NOT_FOUND, "interaction_not_found", &ix),
    }
}

/// End one, and walk the person back out of the world.
async fn end_interaction(State(s): State<Arc<Authored>>, Path(ix): Path<String>) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    let session = rt.interactions.get(&ix, crate::api::now_ms());
    if !rt.interactions.end(&ix) {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    }
    // The body goes with the session. A person who closed the conversation has
    // gone, and a room that still lists them is a room where a character is
    // told it has company that is not there.
    let left = match session {
        Some(was) if was.mode == crate::engine::tools::Mode::Physical => rt.leave_world(&was.body),
        _ => false,
    };
    Json(json!({ "interaction_id": ix, "state": "ended", "left_the_world": left })).into_response()
}

/// Say something to the character, inside a session.
///
/// **The same door everything else goes through.** The line is parsed by
/// [`crate::engine::slash`] and delivered by the scheduler, exactly as
/// `/v1/npc/:nid/pulse` does — so what is said here is a thing that happened to
/// the character rather than a private aside, and the rest of the world sees it
/// the way it sees anything else.
async fn inject(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(ix): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("speaking to a character");
    };
    let now = crate::api::now_ms();
    let Some(session) = rt.interactions.get(&ix, now) else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    // The session says which character; ownership is still checked, because an
    // interaction id is not a capability.
    if let Err(r) = owned(&s, &headers, &session.npc_id.to_string()).await {
        return *r;
    }
    let line = body
        .get("line")
        .or_else(|| body.get("text"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if line.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_line",
            "there is nothing in that to say",
        );
    }
    let parsed = match crate::engine::slash::parse(line) {
        // Named, or the character is told "you says to you" — see
        // [`crate::engine::slash::Parsed::attributed_to`].
        Ok(p) => p.attributed_to(&session.interlocutor.display),
        // A typo is a 400 naming the near miss, never speech — sending `/hrut`
        // to a character as dialogue is the one outcome that looks like it
        // worked.
        Err(e) => return err(StatusCode::BAD_REQUEST, "bad_command", &e.message()),
    };
    let world_ms = s.world_ms(session.npc_id).await;
    let kind = match parsed.kind {
        crate::engine::event::EventKind::Sleep { .. } => crate::engine::event::EventKind::Sleep {
            day: crate::engine::sleep::day_of(world_ms),
        },
        other => other,
    };
    let prose =
        crate::engine::event::Event::new(0, world_ms, parsed.salience, kind.clone()).prose();
    if !rt
        .scheduler
        .deliver(session.npc_id, world_ms, parsed.salience, kind)
    {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "not_awake",
            "that character is not awake, so nothing can reach it",
        );
    }
    rt.interactions.touched(&ix, now);
    Json(json!({ "delivered": true, "prose": prose })).into_response()
}

/// How the person who made these worlds is marked, in the world.
///
/// **Part of the name rather than a field beside it**, because a name is the
/// whole of how anybody is addressed here: threads, rooms and the roster all
/// key on it, and `tell`, `ask` and `message` bind their addressee to a closed
/// set of exactly these strings. A separate "is the creator" flag would have to
/// be carried to every one of those places and rendered into the prose at each,
/// and the first one that forgot would be a character talking to a stranger.
///
/// Carried in the name, it needs no plumbing at all: it is what a character
/// reads when the message arrives, and it is what the grammar offers back when
/// the character answers.
const CREATOR_MARK: &str = "(The Creator)";

/// What a person is called, on a thread with a character.
///
/// **The name the world writes down**, because a thread addresses people the
/// same way a room does and one person must never be two. An account handle
/// (`u_1a2b3c4d`) would read as a stranger in the prose the character is handed
/// — "u_1a2b3c4d messages you" — so the identity's own name is used when the
/// provider gave one, and the handle is the fallback that at least stays
/// stable.
///
/// # Why the role is read here
///
/// This is the one place a person outside the world acquires a name inside it,
/// so it is the only place [`Role::Creator`] can be turned into something the
/// fiction can see. The role is derived from the identity through the same
/// [`Roles::of`] the guard used, rather than passed in — two computations of
/// "is this the creator" could disagree, and the one that decided what a
/// character *reads* would be the one nothing tested.
pub(crate) fn speaking_as(
    id: &web::auth::session::Identity,
    handle: &str,
    roles: &web::auth::Roles,
) -> String {
    let name = match id.name.trim() {
        "" => handle,
        name => name,
    };
    match roles.of(Some(id)).is_creator() {
        true => format!("{name} {CREATOR_MARK}"),
        false => name.to_string(),
    }
}

/// `POST /v1/npc/:nid/message` — say something to a character on its handset.
///
/// Goes onto the same thread the characters use, so the character is told about
/// it by the ordinary sweep and answers with the ordinary `message` act. A
/// person is a party to the conversation rather than an operator poking at one.
async fn post_message(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let text = body
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if text.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_message",
            "a message needs something in it",
        );
    }
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("messaging a character");
    };
    let Some(sent) = rt.message_npc(npc_id, &me, text) else {
        return err(
            StatusCode::CONFLICT,
            "not_in_a_world",
            "that character has no body in a world, so there is nothing to reach it on",
        );
    };
    Json(json!({
        "sent": text,
        "from": me,
        "to": sent.with,
        "waiting_for_them": sent.waiting_for_them,
        "can_reply": sent.can_reply,
    }))
    .into_response()
}

/// `GET /v1/npc/:nid/message` — the conversation so far, oldest first.
async fn get_messages(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let npc_id = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let me = speaking_as(&id, &handle, &s.roles);
    let said = s
        .runtime
        .as_ref()
        .and_then(|rt| rt.messages_with(npc_id, &me));
    let Some((them, messages)) = said else {
        // Not an error: a character with no body has nothing to say on a
        // handset yet, and the console renders an empty conversation.
        return Json(json!({ "messages": [], "in_a_world": false })).into_response();
    };
    let messages: Vec<Value> = messages
        .into_iter()
        .map(|(from, text)| json!({ "from": from, "text": text }))
        .collect();
    Json(json!({
        "messages": messages,
        "in_a_world": true,
        "with": them,
        "as": me,
    }))
    .into_response()
}

/// `POST /v1/world/:wid/posting` — leave words on something in a world.
///
/// **Creator only**, for the reason the channel's write side is: it writes into
/// a world rather than into a character somebody owns, and ownership has
/// nothing to say about that.
///
/// The body names `at` (the room, as `area/node`), `on` (the surface), and
/// `text`. A surface the map never named is stood up on the spot, so a world can
/// grow a board without its map being edited and reloaded — but the *room* must
/// exist, or the board would be writable, unreachable, and invisible.
async fn post_posting(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let field = |k: &str| {
        body.get(k)
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string()
    };
    let (at, on, text) = (field("at"), field("on"), field("text"));
    if at.is_empty() || on.is_empty() || text.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "incomplete",
            "a posting needs `at` (area/node), `on` (what to write on) and `text`",
        );
    }
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("posting into a world");
    };
    let Some(lines) = rt.post_in_world(&wid, &at, &on, &me, &text) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_place",
            "that world is not hosted, or it has no such room — a board in a room that does not \
             exist could never be read",
        );
    };
    Json(json!({
        "posted": text,
        "world": wid,
        "at": at,
        "on": on,
        "by": me,
        "lines": lines,
    }))
    .into_response()
}

/// `GET /v1/world/:wid/channel` — what has been said on a world's open channel.
///
/// Read as the caller, so it shows what the channel has carried since they
/// joined it and not the hours before — the same rule every other member reads
/// it under.
async fn get_channel(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading a world's channel");
    };
    let Some(said) = rt.channel(&wid, &me) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_world",
            "that world is not hosted, so it has no channel",
        );
    };
    let messages: Vec<Value> = said
        .into_iter()
        .map(|(from, text)| json!({ "from": from, "text": text }))
        .collect();
    Json(json!({
        "channel": crate::sim::phone::CHANNEL,
        "world": wid,
        "as": me,
        "messages": messages,
    }))
    .into_response()
}

/// `POST /v1/world/:wid/channel` — say something on a world's open channel.
///
/// **Creator only.** Everything else on this daemon is scoped by ownership: a
/// route reaches the characters the caller owns and no others. This one reaches
/// every character in a world at once, which is not a thing ownership can
/// express — so it is bound to the one role that is *about* standing outside
/// the whole world rather than owning part of it.
async fn post_channel(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let (id, handle) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let text = body
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if text.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_message",
            "a message needs something in it",
        );
    }
    let me = speaking_as(&id, &handle, &s.roles);
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("speaking on a world's channel");
    };
    let Some(heard) = rt.say_on_channel(&wid, &me, text) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_world",
            "that world is not hosted, so it has no channel",
        );
    };
    Json(json!({
        "sent": text,
        "channel": crate::sim::phone::CHANNEL,
        "world": wid,
        "from": me,
        "heard_by": heard,
    }))
    .into_response()
}

/// How often the stream looks for new ticks.
///
/// **Polled off the scheduler's own ring rather than pushed from the decode
/// loop.** A broadcast channel out of `record_act` would be a second path by
/// which an act becomes observable, and the two would drift — the Pulse view
/// already reads this ring, so a session watching the same rows is watching the
/// same truth. The cost is the latency below, against a character that thinks
/// in seconds.
const STREAM_POLL: std::time::Duration = std::time::Duration::from_millis(500);

/// Which tick a console's attachment starts after.
///
/// **`since` is what makes coming back different from starting again.** A
/// conversation outlives the page it is watched from: look at Pulse for a
/// minute and the character goes on acting the whole time. Starting every
/// attachment from the newest tick threw all of that away — you came back to
/// the transcript you left and the minute in between had simply not happened,
/// which is the wrong answer to the one question somebody returning is asking.
///
/// So a console names the last tick it already has and gets what followed.
/// Clamped to `latest`, because a tick from the future — a console that
/// outlived a daemon restart, a hand-typed URL — would otherwise produce a
/// stream that connects and never sends anything, which is indistinguishable
/// from a character that has stopped thinking. Absent or unparseable means a
/// fresh attachment: `latest`, so nobody is replayed a whole afternoon.
///
/// The honest limit is above this function: the scheduler keeps a bounded
/// window, so a console away for longer than that memory gets what is left of
/// the interval rather than all of it.
fn resume_from(since: Option<&str>, latest: u64) -> u64 {
    since
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(latest, |t| t.min(latest))
}

/// `GET /v1/interaction/:ix/stream` — what the character does, as it does it.
///
/// Frames are the console's: `open` once, then `act` per act with `tick` at the
/// close of each. There is no `narration` frame, and its absence is honest —
/// this daemon renders an act *as* its own line rather than producing a
/// separate account afterwards, so a narration frame would be prose nothing
/// wrote.
///
/// `?since=<tick>` resumes an attachment where a console left off — see
/// [`resume_from`], which is the difference between coming back to a
/// conversation and starting a new one.
async fn stream(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(ix): Path<String>,
    Query(q): Query<BTreeMap<String, String>>,
) -> Response {
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("streaming an interaction");
    };
    let now = crate::api::now_ms();
    let Some(session) = rt.interactions.get(&ix, now) else {
        return err(StatusCode::NOT_FOUND, "interaction_not_found", &ix);
    };
    if let Err(r) = owned(&s, &headers, &session.npc_id.to_string()).await {
        return *r;
    }

    let rt = rt.clone();
    let npc_id = session.npc_id;

    // The newest tick this character has taken. Where a fresh console starts,
    // so attaching to a live session does not replay the whole afternoon.
    let latest: u64 = rt
        .scheduler
        .recent(512)
        .iter()
        .filter(|t| t.npc_id == npc_id)
        .map(|t| t.tick)
        .max()
        .unwrap_or(0);

    let mut seen: u64 = resume_from(q.get("since").map(String::as_str), latest);

    let open = Event::default().event("open").data(
        json!({
            "interaction_id": ix,
            "mode": session.mode.as_wire(),
            "resume_from": seen,
        })
        .to_string(),
    );

    // A channel and a task rather than a generator: `async-stream` is not a
    // dependency here, and the task ends by itself when the receiver is dropped
    // — a console that navigates away closes the connection, the send fails,
    // and the loop stops. Nothing has to notice the disconnect separately.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    tokio::spawn(async move {
        if tx.send(Ok(open)).await.is_err() {
            return;
        }
        loop {
            tokio::time::sleep(STREAM_POLL).await;
            let now = crate::api::now_ms();
            // **Holding this stream open is being in the room.** Idle catches
            // the person who walked off; without this it also caught the one
            // who sat and listened, because it was measured from the last line
            // rather than from the last sign of anybody being there. A console
            // that navigates away closes the connection and this loop ends, so
            // the timeout still does its job the moment nobody is watching.
            rt.interactions.attended(&ix, now);
            // The session going quiet ends the stream, so a console left open
            // is not holding a connection against a conversation that is over.
            if rt.interactions.get(&ix, now).is_none() {
                let _ = tx
                    .send(Ok(Event::default()
                        .event("state")
                        .data(json!({ "state": "ended" }).to_string())))
                    .await;
                return;
            }
            let fresh: Vec<_> = rt
                .scheduler
                .recent(512)
                .into_iter()
                .filter(|t| t.npc_id == npc_id && t.tick > seen)
                .collect();
            for t in fresh {
                seen = seen.max(t.tick);
                for (n, act) in t.acts.iter().enumerate() {
                    // `Act::summary` renders `tool — args`; the console wants
                    // the two apart so it can label one and quote the other.
                    let (tool, intent) = match act.split_once(" — ") {
                        Some((a, b)) => (a, b),
                        None => (act.as_str(), ""),
                    };
                    let frame = Event::default().event("act").data(
                        json!({
                            "act_id": format!("{}-{n}", t.tick),
                            "tick": t.tick,
                            "tool": tool,
                            "intent": intent,
                            "world_ms": t.world_ms,
                        })
                        .to_string(),
                    );
                    if tx.send(Ok(frame)).await.is_err() {
                        return;
                    }
                }
                let close = Event::default()
                    .event("tick")
                    .data(json!({ "tick": t.tick, "acts": t.acts.len() }).to_string());
                if tx.send(Ok(close)).await.is_err() {
                    return;
                }
            }
        }
    });
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// The act vocabulary.
///
/// Empty, and `engine_connected: false` beside it so the console can say *why*
/// it is empty. The tools are registered by the engine with the layers each may
/// write, and calibration is a pass it runs; there is no authored catalog in the
/// mind to read one from instead.
async fn calibrate() -> Response {
    no_engine("calibrating tools")
}

/// Real: written by the prose guest, against the world's own setting.
///
/// This was a 503 stub and the console's create page has called it since it was
/// written — the page's fallback text ("generation unavailable — write one
/// yourself") is what an author saw instead. See [`crate::describe`].
async fn gen_description(
    state: State<Arc<crate::api::Authored>>,
    body: Json<crate::describe::DescribeBody>,
) -> Response {
    crate::describe::post_describe(state, body).await
}

/// The same generation, streamed a fragment at a time. See
/// [`crate::describe::post_describe_stream`].
async fn gen_description_stream(
    state: State<Arc<crate::api::Authored>>,
    body: Json<crate::describe::DescribeBody>,
) -> Response {
    crate::describe::post_describe_stream(state, body).await
}

/// Real: named by the prose guest, against the world's own summary. See
/// [`crate::namegen`].
async fn gen_name(
    state: State<Arc<crate::api::Authored>>,
    body: Json<crate::namegen::NameBody>,
) -> Response {
    crate::namegen::post_name(state, body).await
}

async fn gen_attributes() -> Response {
    no_engine("generating attributes")
}

/// Real: drawn by the image guest.
///
/// This answered 503 while `/v1/guest/image` served the same request, so the
/// console's own "draw me an image" route was the one thing on the daemon that
/// could not. The create step needs it: it draws a portrait to show *before*
/// there is a character to address `/v1/npc/:nid/portrait/generate` to.
async fn gen_image(
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
async fn image_models(state: State<Arc<crate::api::Authored>>) -> Response {
    crate::portrait::get_models(state).await
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

#[cfg(test)]
mod speaking_as_tests {
    use super::{speaking_as, CREATOR_MARK};
    use web::auth::session::Identity;
    use web::auth::Roles;

    fn id(email: &str, name: &str) -> Identity {
        Identity {
            provider: "google".into(),
            sub: "g1".into(),
            email: email.into(),
            name: name.into(),
            picture: String::new(),
            exp: 0,
        }
    }

    fn roles(yaml: &str) -> Roles {
        serde_yaml::from_str(yaml).expect("parses")
    }

    /// **The point of the whole flag.** A character reads this string and
    /// answers to it, so the mark has to be in the name itself.
    #[test]
    fn the_creator_is_named_as_such_inside_the_world() {
        let r = roles("creators:\n  - email: me@example.com\n");
        let me = speaking_as(&id("me@example.com", "Johnathan Sharratt"), "u_1", &r);
        assert_eq!(me, format!("Johnathan Sharratt {CREATOR_MARK}"));
        assert!(me.contains("(The Creator)"));
    }

    /// And nobody else is. An ordinary signed-in player carries their own name
    /// and nothing more — a mark everybody had would say nothing.
    #[test]
    fn an_ordinary_person_is_just_their_name() {
        let r = roles("creators:\n  - email: me@example.com\n");
        assert_eq!(
            speaking_as(&id("someone@example.com", "Wren"), "u_2", &r),
            "Wren"
        );
        // Nor is an admin, who is trusted with the files and is still not the
        // person the fiction was made by.
        let r = roles("admins:\n  - email: boss@example.com\n");
        assert_eq!(
            speaking_as(&id("boss@example.com", "Boss"), "u_3", &r),
            "Boss"
        );
    }

    /// The handle is the fallback when the provider gave no display name, and
    /// the mark still lands — a creator whose gateway dropped the name header
    /// must not silently become an anonymous stranger to its own cast.
    #[test]
    fn a_creator_with_no_display_name_still_carries_the_mark() {
        let r = roles("creators:\n  - email: me@example.com\n");
        let me = speaking_as(&id("me@example.com", ""), "u_1a2b3c4d", &r);
        assert_eq!(me, format!("u_1a2b3c4d {CREATOR_MARK}"));
    }
}

#[cfg(test)]
mod tests {
    use super::resume_from;

    /// The ordinary case: nobody names a tick, so the attachment starts at the
    /// newest one and does not replay the character's whole afternoon.
    #[test]
    fn a_fresh_attachment_starts_at_the_newest_tick() {
        assert_eq!(resume_from(None, 400), 400);
    }

    /// **The point of the parameter.** A console that has already read up to
    /// tick 380 gets 381 onwards, which is what it missed while its reader was
    /// looking at another page.
    #[test]
    fn a_returning_console_resumes_where_it_left_off() {
        assert_eq!(resume_from(Some("380"), 400), 380);
    }

    /// **A tick from the future must not stall the stream.** Unclamped, `seen`
    /// would start above every tick the character has taken and the loop would
    /// send nothing at all — a connection that opens and stays silent, which is
    /// indistinguishable from a mind that has stopped thinking. A daemon
    /// restart resets tick numbering, so a console outliving one arrives here
    /// with exactly that.
    #[test]
    fn a_tick_from_the_future_falls_back_to_a_fresh_attachment() {
        assert_eq!(resume_from(Some("9000"), 400), 400);
    }

    /// Junk in the query string is not a reason to serve a broken stream.
    #[test]
    fn an_unparseable_since_is_a_fresh_attachment() {
        for junk in ["", "abc", "-5", "3.5", "١٢٣"] {
            assert_eq!(resume_from(Some(junk), 400), 400, "since={junk:?}");
        }
    }

    /// A character that has never ticked has no history to resume into, and
    /// asking for one must not underflow into replaying from zero for ever.
    #[test]
    fn a_character_that_has_never_ticked_resumes_at_nothing() {
        assert_eq!(resume_from(Some("12"), 0), 0);
        assert_eq!(resume_from(None, 0), 0);
    }
}
