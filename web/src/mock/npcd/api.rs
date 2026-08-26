//! The `/v1/*` surface, mocked.
//!
//! Route shapes and response bodies follow `docs/npc_api_gui_design.md` Part B.
//!
//! This router is **API only** — it serves no HTML and knows nothing about the
//! console, even though it is compiled beside it. Everything the GUI does, it
//! does by calling these routes; a path that looks like an API route and isn't
//! returns a JSON 404 rather than falling through to a page. That is the
//! API-first posture holding even where the temptation to shortcut is largest,
//! and it is why every screen is reachable by anything that can speak HTTP.

use axum::{
    body::Body,
    extract::{Path, Query, Request},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{delete, get, post, put},
    Json, Router,
};
use futures::stream::{self, Stream};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Duration;

use super::{data, schema, turns, ws};

pub fn router() -> Router {
    Router::new()
        // status / telemetry
        .route("/v1/status", get(|| async { Json(data::status()) }))
        .route("/v1/telemetry", get(|| async { Json(data::telemetry()) }))
        // auth
        .route("/v1/auth/providers", get(providers))
        .route("/v1/me", get(|| async { Json(data::user()) }))
        .route("/v1/me/profile", get(profile).put(put_profile))
        .route("/v1/auth/logout", post(|| async { StatusCode::NO_CONTENT }))
        // npcs
        .route("/v1/npc", get(list_npcs).post(create_npc))
        .route("/v1/npc/:id", get(get_npc).patch(patch_npc))
        .route("/v1/npc/:id", delete(|| async { StatusCode::NO_CONTENT }))
        .route("/v1/npc/:id/tags", put(ok))
        .route("/v1/npc/:id/hidden", put(ok))
        .route("/v1/npc/:id/perceive", post(perceive))
        .route("/v1/npc/:id/beliefs", get(|Path(id): Path<String>| async move {
            Json(json!({ "beliefs": data::beliefs(&id) }))
        }))
        .route("/v1/npc/:id/beliefs/:bid", put(ok).delete(|| async { StatusCode::NO_CONTENT }))
        .route("/v1/npc/:id/relationships", get(|Path(id): Path<String>| async move {
            Json(json!({ "relationships": data::relationships(&id) }))
        }))
        .route("/v1/npc/:id/relationships/:eid", put(ok))
        .route("/v1/npc/:id/agency", get(|Path(id): Path<String>| async move {
            Json(json!({ "agency": data::agency(&id) }))
        }))
        .route("/v1/npc/:id/memory", get(memory))
        .route("/v1/npc/:id/modulation", get(modulation).put(ok))
        .route("/v1/npc/:id/substrate", get(|Path(id): Path<String>| async move {
            Json(data::layer_summary(&id))
        }))
        .route("/v1/npc/:id/substrate/layer/:name", get(layer))
        .route("/v1/npc/:id/projection", get(projection_latest))
        .route("/v1/npc/:id/projection/:tick", get(projection_at))
        .route("/v1/npc/:id/monitor", get(monitor))
        .route("/v1/npc/:id/environment", get(|Path(id): Path<String>| async move {
            Json(data::environment(&id))
        }))
        .route("/v1/npc/:id/environment", put(ok))
        .route("/v1/npc/:id/environment/inject", post(ok))
        .route("/v1/npc/:id/interaction", get(list_ix).post(open_ix))
        // interactions
        .route("/v1/interaction/:ix", get(get_ix).delete(|| async { StatusCode::NO_CONTENT }))
        .route("/v1/interaction/:ix/inject", post(ok))
        .route("/v1/interaction/:ix/stream", get(stream_ix))
        // worlds / archetypes
        .route("/v1/world", get(|| async { Json(json!({ "worlds": data::worlds() })) }).post(ok))
        .route("/v1/world/:wid", get(get_world).put(ok))
        .route("/v1/world/:wid/time", put(ok))
        .route("/v1/archetype", get(|| async { Json(json!({ "archetypes": data::archetypes() })) }))
        .route("/v1/archetype/:aid", get(get_archetype).put(ok))
        // schema: layers + section collections (a world IS its schema)

        .route("/v1/schema/layers", get(|| async { Json(schema::layers()) }))

        .route("/v1/world/:wid/collections", get(|| async { Json(schema::world_collections()) }))

        .route("/v1/archetype/:aid/collections", get(|| async { Json(schema::archetype_collections()) }))

        // Push streams. The panes they feed do not poll — `/ws/logs` replays
        // the backlog on connect, so there is no seed-then-subscribe gap.
        .route("/ws/logs", get(ws::logs))
        .route("/ws/events", get(ws::events))
        // One turn's body + its K/V segment vector, and the live retrieval probe.
        .route("/v1/npc/:id/substrate/turn/:layer/:turn", get(get_turn))
        .route("/v1/npc/:id/project", post(project))

        // tools / commands / generation / images
        .route("/v1/tools", get(|| async { Json(data::tools()) }))
        .route("/v1/tools/calibrate", post(|| async {
            Json(json!({ "job_id": "job_cal_1", "tools": ["open_gate"] }))
        }))
        .route("/v1/commands", get(|| async { Json(data::commands()) }))
        .route("/v1/generate/description", post(gen_description))
        .route("/v1/generate/attributes", post(gen_attributes))
        .route("/v1/image/generate", post(gen_image))
        .route("/v1/image/models", get(image_models))
        .route("/v1/image/queue", get(image_queue))
        .fallback(route_not_found)
}

// ── small helpers ────────────────────────────────────────────────────────────

async fn ok() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "ok": true })))
}

fn not_found(code: &str, detail: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": code, "detail": detail, "field": null })),
    )
        .into_response()
}

// ── handlers ─────────────────────────────────────────────────────────────────

async fn providers() -> impl IntoResponse {
    Json(json!({ "providers": [
        { "id": "google", "display": "Google", "icon": "google" },
        { "id": "github", "display": "GitHub", "icon": "github" }
    ]}))
}

async fn profile() -> impl IntoResponse {
    Json(data::user()["profile"].clone())
}

async fn put_profile(Json(body): Json<Value>) -> impl IntoResponse {
    let mut p = data::user()["profile"].clone();
    if let (Some(dst), Some(src)) = (p.as_object_mut(), body.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
        dst.insert("revision".into(), json!(4));
    }
    Json(p)
}

async fn list_npcs(Query(q): Query<HashMap<String, String>>) -> impl IntoResponse {
    let tag = q
        .get("tag")
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty());
    let state = q
        .get("state")
        .filter(|s| !s.is_empty() && s.as_str() != "any");
    let world = q.get("world_id").filter(|s| !s.is_empty());

    let items: Vec<Value> = data::npcs()
        .into_iter()
        .filter(|n| {
            // §8.3 — the whole discretion rule, in one predicate:
            // without a tag filter, hidden NPCs are omitted; WITH one, they
            // match like anything else and are returned indistinguishably.
            let hidden = n["hidden"].as_bool().unwrap_or(false);
            let tag_hit = tag.as_ref().map(|t| {
                n["tags"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .any(|x| x.as_str().unwrap_or("").to_lowercase().contains(t.as_str()))
                    })
                    .unwrap_or(false)
            });
            match (&tag_hit, hidden) {
                (Some(false), _) => return false,
                (None, true) => return false,
                _ => {}
            }
            if let Some(s) = state {
                if n["state"].as_str() != Some(s.as_str()) {
                    return false;
                }
            }
            if let Some(w) = world {
                if n["world_id"].as_str() != Some(w.as_str()) {
                    return false;
                }
            }
            true
        })
        .collect();

    Json(json!({ "items": items, "next_cursor": null, "has_more": false }))
}

async fn create_npc(Json(body): Json<Value>) -> impl IntoResponse {
    let mut n = data::npcs()[0].clone();
    if let Some(o) = n.as_object_mut() {
        o.insert("npc_id".into(), json!(data::new_interaction_id()));
        if let Some(name) = body.get("name") {
            o.insert("name".into(), name.clone());
        }
        if let Some(w) = body.get("world_id") {
            o.insert("world_id".into(), w.clone());
        }
        o.insert("tags".into(), json!([]));
        o.insert("hidden".into(), json!(false));
    }
    (StatusCode::CREATED, Json(n))
}

async fn get_npc(Path(id): Path<String>) -> Response {
    match data::npcs().into_iter().find(|n| n["npc_id"] == json!(id)) {
        Some(n) => Json(n).into_response(),
        None => not_found("npc_not_found", &format!("no NPC with id {id}")),
    }
}

async fn patch_npc(Path(id): Path<String>, Json(body): Json<Value>) -> Response {
    match data::npcs().into_iter().find(|n| n["npc_id"] == json!(id)) {
        Some(mut n) => {
            if let (Some(o), Some(src)) = (n.as_object_mut(), body.as_object()) {
                for (k, v) in src {
                    o.insert(k.clone(), v.clone());
                }
            }
            Json(n).into_response()
        }
        None => not_found("npc_not_found", &format!("no NPC with id {id}")),
    }
}

async fn perceive(Json(body): Json<Value>) -> impl IntoResponse {
    let n = body["events"].as_array().map(|a| a.len()).unwrap_or(0);
    (
        StatusCode::ACCEPTED,
        Json(json!({ "accepted": n, "tick_scheduled": n > 0, "preempted": false })),
    )
}

async fn memory(Path(id): Path<String>) -> impl IntoResponse {
    let _ = id;
    let items: Vec<Value> = (0..24)
        .map(|i| {
            json!({
                "turn": 4412 - i,
                "world_ms": data::world_ms() - i as u64 * 3_600_000,
                "text": MEMORY_LINES[(i as usize) % MEMORY_LINES.len()],
                "tokens": 120 + (i as u64 % 7) * 30
            })
        })
        .collect();
    Json(json!({ "items": items, "next_cursor": "eyJ0IjoyNzN9", "has_more": true }))
}

const MEMORY_LINES: [&str; 6] = [
    "The mill road washed out; the crossing moved a mile north and nobody told the garrison.",
    "Hess countermanded the rotation twice in one week, then denied the second order.",
    "Ilse would not take coin for the tobacco, which meant she wanted something later.",
    "A horn from the eastern slope, twice — the signal for ground given, not for contact.",
    "The recruit from the coast asked why the fallback was never written down.",
    "Winter came early enough that the northern road froze before it flooded.",
];

async fn modulation() -> impl IntoResponse {
    Json(json!({ "affect": -0.2, "threat": 0.66, "curiosity": 0.3 }))
}

async fn layer(Path((id, name)): Path<(String, String)>) -> impl IntoResponse {
    let _ = id;
    let items: Vec<Value> = (0..14)
        .map(|i| {
            json!({
                "turn": 200 - i,
                "world_ms": data::world_ms() - i as u64 * 600_000,
                "score": (0.95 - (i as f64) * 0.05).max(0.05),
                "tokens": 90 + (i as u64 % 5) * 40,
                "kind": if name == "action" { "act" } else { "text" },
                "preview": layer_preview(&name, i)
            })
        })
        .collect();
    Json(json!({ "layer": name, "items": items, "next_cursor": null, "has_more": false }))
}

fn layer_preview(layer: &str, i: usize) -> String {
    match layer {
        "perception" => [
            "A horn, twice, from the eastern slope.",
            "Wind off the ridge; the light going amber.",
            "The line east of the mill gives ground.",
            "Movement in the treeline — two, maybe three.",
        ][i % 4]
            .into(),
        "action" => [
            "speak → \"Quiet, so far.\"",
            "face → east",
            "move_to → ridge_east",
            "observe → eastern_line",
        ][i % 4]
            .into(),
        "memory" => MEMORY_LINES[i % MEMORY_LINES.len()].into(),
        "world" => [
            "The crown's courier has not come in eleven days.",
            "Tolls on the north road doubled after the thaw.",
        ][i % 2]
            .into(),
        "environment" => [
            "The light goes amber and the wind drops.",
            "Rain starts, fine and cold, from the west.",
        ][i % 2]
            .into(),
        _ => "…".into(),
    }
}

async fn projection_latest() -> impl IntoResponse {
    Json(data::projection(412))
}

async fn projection_at(Path((_id, tick)): Path<(String, u64)>) -> impl IntoResponse {
    Json(data::projection(tick))
}

async fn monitor(Query(q): Query<HashMap<String, String>>) -> impl IntoResponse {
    let w = q
        .get("window")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100usize);
    Json(data::monitor(w.clamp(10, 400)))
}

async fn list_ix(Path(id): Path<String>) -> impl IntoResponse {
    Json(json!({ "interactions": data::interactions(&id) }))
}

async fn open_ix(Path(id): Path<String>, Json(body): Json<Value>) -> impl IntoResponse {
    let mode = body["mode"].as_str().unwrap_or("physical").to_string();
    let ix = data::new_interaction_id();
    (
        StatusCode::CREATED,
        Json(data::interaction(&ix, &id, &mode)),
    )
}

async fn get_ix(Path(ix): Path<String>) -> impl IntoResponse {
    Json(data::interaction(&ix, "10237749914772934281", "physical"))
}

/// The two-latency stream (§19): `act` frames live, `narration` at tick close.
/// The mock replays a canned script on a timer so the console behaves like the
/// real thing — acts arriving ahead of the prose that explains them.
async fn stream_ix(Path(ix): Path<String>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut frames: Vec<Event> = vec![Event::default().event("open").data(
        json!({ "interaction_id": ix, "mode": "physical", "resume_from": null }).to_string(),
    )];

    for f in data::script() {
        let name = match f.get("kind").and_then(|k| k.as_str()) {
            Some("tick") => "tick",
            Some("narration") => "narration",
            _ => "act",
        };
        frames.push(Event::default().event(name).data(f.to_string()));
        if name == "act" {
            frames.push(
                Event::default()
                    .event("act_rendered")
                    .data(json!({ "act_id": f["act_id"], "rendered": f["rendered"] }).to_string()),
            );
        }
    }

    let s = stream::iter(frames).then(|e| async move {
        tokio::time::sleep(Duration::from_millis(700)).await;
        Ok(e)
    });
    use futures::StreamExt;
    Sse::new(s).keep_alive(axum::response::sse::KeepAlive::default())
}

async fn get_world(Path(wid): Path<String>) -> Response {
    match data::worlds()
        .into_iter()
        .find(|w| w["world_id"] == json!(wid))
    {
        Some(w) => Json(w).into_response(),
        None => not_found("world_not_found", &format!("no world {wid}")),
    }
}

async fn get_archetype(Path(aid): Path<String>) -> Response {
    match data::archetypes()
        .into_iter()
        .find(|a| a["archetype_id"] == json!(aid))
    {
        Some(a) => Json(a).into_response(),
        None => not_found("archetype_not_found", &format!("no archetype {aid}")),
    }
}

async fn gen_description() -> impl IntoResponse {
    const P: [&str; 4] = [
        "Fifty-three, a former staff sergeant who now runs the night shift on a loading dock. \
         Precise about time to the point of rudeness. Comfortable giving orders, uneasy in \
         conversations with no clear purpose. Keeps a folding knife he doesn't use.",
        "Early forties, teaches secondary maths and referees on weekends. Explains things in \
         numbered steps whether or not you asked. Cannot let a wrong claim stand, which has \
         cost him two friendships.",
        "Sixty-eight, retired from a job she will not name. Watches the street from a first-floor \
         window and knows the delivery schedule of every van on it. Kind to strangers, guarded \
         with neighbours.",
        "Twenty-nine, works nights in a hospital laundry and paints in the mornings. Speaks \
         quietly and rarely first. Notices what people do with their hands.",
    ];
    let idx = (data::now_ms() / 1000) as usize % P.len();
    Json(json!({ "description": P[idx], "seed": 88213 + idx as u64 }))
}

async fn gen_attributes() -> impl IntoResponse {
    Json(json!({
        "beliefs": data::beliefs("x"),
        "relationships": data::relationships("x"),
        "agency": data::agency("x")
    }))
}

async fn gen_image() -> impl IntoResponse {
    (
        StatusCode::ACCEPTED,
        Json(json!({
            "job_id": "job_img_1", "kind": "image", "state": "queued",
            "progress": 0.0, "queue_position": 2, "eta_secs": null, "result": null, "error": null
        })),
    )
}

async fn image_models() -> impl IntoResponse {
    Json(json!({ "models": [
        { "id": "sdxl-turbo", "display": "SDXL Turbo", "vram_gib": 8.0, "loaded": false, "default": true },
        { "id": "sd15",       "display": "Stable Diffusion 1.5", "vram_gib": 2.8, "loaded": false },
        { "id": "wuerstchen", "display": "Würstchen", "vram_gib": 3.6, "loaded": false }
    ]}))
}

async fn image_queue() -> impl IntoResponse {
    Json(json!({ "depth": 2, "position": 1, "state": "waiting_for_vram", "next_run_eta": null }))
}

// ── fallback ────────────────────────────────────────────────────────────────

/// Every unmatched path this router is given is a JSON 404. It never returns
/// HTML — the console is served by the `web` crate in front of it, and a
/// request that reached here was routed by an API prefix. An API path must
/// always fail like an API path.
async fn route_not_found(req: Request<Body>) -> Response {
    not_found("route_not_found", req.uri().path())
}

async fn get_turn(Path((id, layer, turn)): Path<(String, String, u64)>) -> impl IntoResponse {
    Json(turns::turn(&id, &layer, turn))
}

/// The live retrieval probe (§36): prefill a hypothetical message and report
/// what the gather would select for it, scored.
async fn project(Path(id): Path<String>, Json(body): Json<Value>) -> impl IntoResponse {
    Json(turns::project(&id, body["text"].as_str().unwrap_or("")))
}
