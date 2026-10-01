//! The phone's routes on `http://local/phone` — the personal messaging surface
//! (effector design §7.2, C.27).
//!
//! Unlike the situated namespaces ([`crate::effector::station`], [`crate::
//! effector::lift`]), the phone is the character's *own*: it is reachable
//! wherever the body stands, because a handset is carried rather than placed
//! (§7). So there is no instance id to resolve against the map — the caller
//! resolves straight to its body ([`Local::body_of`]) and the phone answers from
//! the threads that body is on.
//!
//! Four routes, mounted by [`crate::effector::router`]:
//!
//! - `GET /phone` — the character's thread listing: every open conversation this
//!   body is on, named the way the character names it, with who is on it and
//!   what is waiting. One read of the world's threads under one lock; an empty
//!   list for a body with no threads (a handset that has reached nobody, or none
//!   at all).
//! - `OPTIONS /phone` — the schema (§5.1): one `POST` verb per phone act, each
//!   body built from the act's own `params` by the same [`body_schema`] the
//!   station routes use, so what the schema advertises is exactly what a `POST`
//!   accepts. `sign_off` is listed apart because it is addressed *per-thread*.
//! - `POST /phone/:verb` — act. The verb is the world act of the same name
//!   (`message`/`invite`/`open_group`/`reach_out`); the body's declared params
//!   are read into the act's `args` and it is run through the real dispatch
//!   ([`body::perform`]), mapped by [`enact_response`]. So the acts' own
//!   gating — an empty set of threads or contacts is the act's refusal, `409` —
//!   is the route's gating too, exactly as the lift and station routes work.
//!   `send_image` is **not** among these verbs: it is performed above the body
//!   layer (see [`VERBS`]), so it is not routable through [`body::perform`] and
//!   is deferred rather than mounted here.
//! - `POST /phone/:thread/sign_off` — leave one conversation. The thread from the
//!   path is what reaches the `sign_off` act as the conversation being left; the
//!   body carries only the parting `intent`.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde_json::{json, Map, Value};

use crate::api::err;
use crate::effector::auth::DeviceCaller;
use crate::effector::enact_route::enact_response;
use crate::effector::router::Local;
use crate::effector::station::body_schema;
use crate::engine::act::Act;
use crate::engine::body;
use crate::engine::tools::{self, Tool};
use crate::sim::phone::{Kind, Thread};

/// The phone verbs addressed at the phone itself — `POST /phone/<verb>`.
///
/// Each is the name of the world act it synthesises, so the verb *is* the tool
/// name (`Tool::name`); the reconstruction the station routes need (namespace +
/// verb) does not apply here, because the phone acts carry no namespace prefix.
/// `sign_off` is not in this list: it is addressed per-thread and has its own
/// route.
///
/// `send_image` is deliberately absent. It is **not** a body act — it is not in
/// [`crate::engine::body::is_of_the_body`], [`crate::engine::enact::is_mine`]
/// omits it, and [`crate::engine::enact::perform`] falls through to
/// [`crate::engine::body::Outcome::NotOfTheBody`] for it — so routing it through
/// [`body::perform`] would map to a `500` on every call. The engine performs
/// send_image above the body layer (the image-guest / interaction path), which
/// this route cannot reach; a later cut wires that path, and `/phone/send_image`
/// is deferred until then rather than advertised as a route that only errors.
const VERBS: &[&str] = &["message", "invite", "open_group", "reach_out"];

/// `GET /phone` — the character's thread listing.
///
/// Every open conversation this body is on, from what the sim actually stores
/// ([`crate::sim::phone::Threads`]). Read under one lock ([`Hosted::with_both`]),
/// because the thread names are the sim's and the body's own name is the world's,
/// and taking the lock twice would let the two disagree. An empty list for a
/// body with no threads.
pub async fn threads(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
) -> Response {
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_phone();
    };
    let listing = hosted.with_both(|w, s| {
        // The body's own name — threads are keyed and named by the name the world
        // writes down, the same address a character uses in a room.
        let me = w.actor(&body).map(|a| a.name.clone()).unwrap_or_default();
        let threads: Vec<Value> = s
            .threads
            .of(&me)
            .into_iter()
            .map(|t| thread_summary(t, &me))
            .collect();
        json!({ "threads": threads })
    });
    Json(listing).into_response()
}

/// One thread, as the character reads it: what it is called *to this member*, who
/// else is on it, whether it is a group, how much is unread, and the last thing
/// said.
fn thread_summary(thread: &Thread, me: &str) -> Value {
    let last = thread
        .messages
        .last()
        .map(|m| json!({ "from": m.from, "intent": m.intent }));
    json!({
        "id": thread.id,
        "with": thread.as_named_to(me),
        "kind": match thread.kind() {
            Kind::Direct => "direct",
            Kind::Group => "group",
        },
        "members": thread.others(me),
        "unread": thread.unread_for(me),
        "last": last,
    })
}

/// `OPTIONS /phone` — the resource schema (effector design §5.1, C.27).
///
/// One `POST` verb per phone act, each body built from the act's own `params` by
/// [`body_schema`] — the identical raw shape the station routes advertise, never
/// invented here. `sign_off` is listed apart: it is addressed per-thread
/// (`POST /phone/<thread>/sign_off`), so its thread comes from the path and its
/// body carries only the parting `intent`.
pub async fn schema(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
) -> Response {
    let Some((_hosted, _body)) = local.body_of(&caller) else {
        return no_phone();
    };
    let mut verbs = Map::new();
    for name in VERBS {
        if let Some(tool) = tools::by_name(name) {
            verbs.insert((*name).to_string(), json!({ "body": body_schema(tool) }));
        }
    }
    // `sign_off` acts on a conversation, so it is addressed per-thread and its
    // `to` (the thread) is the path segment rather than a body field — the schema
    // says so, and its body drops `to`, keeping only the parting `intent`.
    if let Some(tool) = tools::by_name("sign_off") {
        verbs.insert(
            "sign_off".to_string(),
            json!({
                "body": body_schema_excluding(tool, "to"),
                "addressed": "per-thread",
                "path": "phone/<thread>/sign_off",
            }),
        );
    }
    Json(json!({
        "id": "phone",
        "summary": "Your phone: your conversations, and the people you can reach.",
        "methods": {
            "GET": { "returns": { "threads": "array" } },
            "POST": { "verbs": verbs },
        },
    }))
    .into_response()
}

/// The body schema of an act with one param left out — for `sign_off`, whose
/// `to` comes from the path, not the body. The rest is [`body_schema`]'s rule:
/// each param a property typed by its `ty`, the required ones listed.
fn body_schema_excluding(tool: &Tool, skip: &str) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for p in tool.params {
        if p.name == skip {
            continue;
        }
        properties.insert(p.name.to_string(), json!({ "type": p.ty }));
        if p.required {
            required.push(Value::String(p.name.to_string()));
        }
    }
    json!({ "type": "object", "properties": properties, "required": required })
}

/// `POST /phone/:verb` — act on the phone.
///
/// The verb must be one the phone affords ([`VERBS`]), or it is not something
/// that can be done here (`404`). Only the act's declared params are carried
/// through, each as the body gave it, and the act is run through the real
/// dispatch — so the act's own gating (no thread, no contact) is the route's,
/// mapped to `409` by [`enact_response`].
pub async fn invoke(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(verb): Path<String>,
    payload: Option<Json<Value>>,
) -> Response {
    if !VERBS.contains(&verb.as_str()) {
        return unknown_verb(&verb);
    }
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_phone();
    };
    // A verb in `VERBS` is a catalogue act by construction — the list is exactly
    // the phone acts' own names.
    let tool = tools::by_name(&verb).expect("a phone verb is a catalogue act");
    let act = Act {
        tool: tool.name,
        args: args_from(tool, payload),
    };
    enact_response(body::perform(&hosted, &body, &act))
}

/// `POST /phone/:thread/sign_off` — leave one conversation.
///
/// The thread from the path is what reaches the `sign_off` act as `to`, the
/// conversation being left; the body carries only the parting `intent`. The act
/// resolves the thread by the name the character calls it *or* by its id
/// ([`crate::sim::phone::Threads::leave`]), so the id a listing hands back works
/// here directly.
pub async fn sign_off(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(thread): Path<String>,
    payload: Option<Json<Value>>,
) -> Response {
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_phone();
    };
    let tool = tools::by_name("sign_off").expect("sign_off is a catalogue act");
    let mut args = Map::new();
    // The thread is the conversation being left — it reaches the act as `to`.
    args.insert("to".to_string(), Value::String(thread));
    if let Some(Json(Value::Object(fields))) = payload {
        if let Some(intent) = fields.get("intent") {
            args.insert("intent".to_string(), intent.clone());
        }
    }
    let act = Act {
        tool: tool.name,
        args,
    };
    enact_response(body::perform(&hosted, &body, &act))
}

/// Read the act's declared params out of the JSON body, each as the body gave
/// it — the same latitude the station routes give, so the handler reads strings
/// off the map exactly as it does when the act is reached any other way.
fn args_from(tool: &Tool, payload: Option<Json<Value>>) -> Map<String, Value> {
    let mut args = Map::new();
    if let Some(Json(Value::Object(fields))) = payload {
        for p in tool.params {
            if let Some(value) = fields.get(p.name) {
                args.insert(p.name.to_string(), value.clone());
            }
        }
    }
    args
}

/// The refusal for a caller whose body cannot be resolved to a hosted world —
/// there is no phone in the hand of a body that is nowhere.
fn no_phone() -> Response {
    err(
        StatusCode::NOT_FOUND,
        "no_phone",
        "you have no body in a running world, so you have no phone",
    )
}

/// The `404` for a verb the phone does not afford.
fn unknown_verb(verb: &str) -> Response {
    err(
        StatusCode::NOT_FOUND,
        "unknown_verb",
        &format!("`{verb}` is not something you can do on your phone"),
    )
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
    use axum::http::{Request, StatusCode};
    use npc_map::world::Where;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use crate::effector::router::{router, Local};
    use crate::effector::token::{Scope, Tokens};
    use crate::engine::runtime::Runtime;
    use crate::mind::Mind;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
    const WORLD: &str = "creators-vault";

    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-phone-ut-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// A held-still vault with the device installed and one body placed in it.
    fn daemon(npc_id: u64) -> (Arc<Runtime>, Arc<Tokens>) {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
        rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
        rt.hold_world(WORLD, true);
        let tokens = Arc::new(Tokens::load(tmp()).expect("a fresh token store"));
        rt.set_tokens(tokens.clone());

        let body = Runtime::body_id(npc_id);
        let world = rt.hosted.get(WORLD).expect("hosted");
        world.with(|w| {
            w.enter(
                &body,
                format!("Maker-{npc_id:02}"),
                Where::new("vault-casting", "band-one"),
            )
            .expect("a real room");
        });
        rt.bodies.bind(npc_id, WORLD, &body).expect("bound");
        (rt, tokens)
    }

    async fn send(
        rt: &Arc<Runtime>,
        tokens: &Arc<Tokens>,
        npc_id: u64,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let token = tokens.mint(npc_id, Scope::AsNpc).expect("minted");
        let app = router(Local::new(tokens.clone(), rt));
        let builder = Request::builder()
            .method(method)
            .uri(path)
            .header(AUTHORIZATION, format!("Bearer {token}"));
        let req = match body {
            Some(v) => builder
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&v).unwrap()))
                .unwrap(),
            None => builder.body(Body::empty()).unwrap(),
        };
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn verb_keys(schema: &Value) -> Vec<String> {
        let mut names: Vec<String> = schema["methods"]["POST"]["verbs"]
            .as_object()
            .expect("a verbs object")
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// **`OPTIONS /phone` advertises the routable verbs only — `send_image` is
    /// gone.** It is not a body act, so a route through `body::perform` would only
    /// `500`; the schema must not offer it.
    #[tokio::test]
    async fn the_schema_advertises_the_routable_verbs_and_not_send_image() {
        let (rt, tokens) = daemon(40);
        let (status, schema) = send(&rt, &tokens, 40, "OPTIONS", "/phone", None).await;
        assert_eq!(status, StatusCode::OK, "{schema}");
        assert_eq!(
            verb_keys(&schema),
            ["invite", "message", "open_group", "reach_out", "sign_off"],
            "{schema}"
        );
        assert!(
            !verb_keys(&schema).contains(&"send_image".to_string()),
            "send_image must not be advertised: {schema}"
        );
        // Each advertised verb carries a raw body schema built from its params.
        assert_eq!(
            schema["methods"]["POST"]["verbs"]["message"]["body"],
            json!({
                "type": "object",
                "properties": { "to": { "type": "string" }, "intent": { "type": "string" } },
                "required": ["to", "intent"]
            }),
            "{schema}"
        );
    }

    /// **Every advertised verb is routable — it runs the real act and never
    /// `500`s.** Whether the act does its thing (`200`) or refuses (`409`) is the
    /// world's call; what matters here is that none falls through to
    /// `NotOfTheBody` (`500`) or is unknown (`404`), which is exactly what
    /// `send_image` did before it was removed.
    #[tokio::test]
    async fn every_advertised_verb_runs_the_real_act() {
        let (rt, tokens) = daemon(41);
        let cases = [
            (
                "message",
                json!({ "to": "the channel", "intent": "where are you" }),
            ),
            ("invite", json!({ "to": "the channel", "who": "Maker-99" })),
            (
                "open_group",
                json!({ "called": "the muster", "with": "Maker-99" }),
            ),
            (
                "reach_out",
                json!({ "to": "Maker-99", "intent": "found you" }),
            ),
        ];
        for (verb, body) in cases {
            let (status, out) = send(
                &rt,
                &tokens,
                41,
                "POST",
                &format!("/phone/{verb}"),
                Some(body),
            )
            .await;
            assert!(
                status == StatusCode::OK || status == StatusCode::CONFLICT,
                "`{verb}` did not run the real act (status {status}): {out}"
            );
            assert_ne!(
                status,
                StatusCode::INTERNAL_SERVER_ERROR,
                "`{verb}` fell through to NotOfTheBody: {out}"
            );
        }
    }

    /// **`send_image` is not routable — it is a `404`, not a `500`.** Removed from
    /// `VERBS`, the phone does not afford it, so it is refused before any act is
    /// synthesised rather than synthesising an act the body cannot perform.
    #[tokio::test]
    async fn send_image_is_not_routable_and_is_a_four_oh_four() {
        let (rt, tokens) = daemon(42);
        let (status, out) = send(
            &rt,
            &tokens,
            42,
            "POST",
            "/phone/send_image",
            Some(json!({ "to": "the channel" })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{out}");
        assert_eq!(out["error"], json!("unknown_verb"), "{out}");
    }
}
