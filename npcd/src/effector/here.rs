//! `http://local/here` — the world-state acts as routes (effector design
//! Appendix G).
//!
//! The generic station mechanism ([`crate::effector::station`]) mounts only the
//! acts an authored part names in its [`Tool::at`]. The world-state half of the
//! catalogue — `claim`/`release`/`operate`/`give`/`equip`/`use`/`recall`/
//! `post_notice`/`record_verdict`/`read`/`scan`/`gather`/`engage` — is not
//! part-bound: its availability is a fact about *where a body stands and what it
//! carries*, not "at this part". This is where those acts become a route.
//!
//! # It is personal, and name-addressed — not instance-addressed
//!
//! Like `/phone` and `/self`, `/here` is the character's *own* surface: it is
//! reachable wherever the body stands (§7.2), so it carries no instance id — the
//! caller resolves straight to its body ([`Local::body_of`]). And its acts name
//! their targets **by the name the world writes down**, resolved against the live
//! `Choices` sets, because that is how the world model actually works: "the blast
//! door" is a [`crate::sim`] device keyed by place, not a map part with an
//! instance id, and "close the longest silence" is a claimable subject with no
//! placement at all. So the target is a field in the body, drawn from an
//! enumerated set the schema advertises — not a segment in the URL.
//!
//! # OPTIONS is the live `Choices` set, which is the whole point
//!
//! [`specs_within`](tools::specs_within) is the one computation that decides, for
//! a body standing here, which acts are reachable and — for each enumerated
//! argument — exactly which values it may take (who is here, what is claimable,
//! what a door can be set to). `/here` renders that computation directly:
//!
//! - `GET /here` — the acts available where the body stands, each with its
//!   one-line description. Exactly the world-state acts [`specs_within`] admits
//!   this moment, so an act that cannot succeed (nothing to claim, no fight to
//!   `engage`) is absent rather than advertised-and-refused — the same discipline
//!   the grammar itself follows.
//! - `OPTIONS /here` — the schema (§5.1): one `POST` verb per available act, each
//!   body a JSON schema whose enumerable properties carry their **live** enum
//!   from world state and whose free properties are plain strings. What the
//!   schema advertises for `what`/`on`/`mode` is exactly what the grammar would
//!   let a character emit, because both are built from the same [`specs_within`].
//! - `POST /here/:verb` — act. The verb must be one `/here` owns and the act's
//!   declared params are read from the body and run through the real dispatch
//!   ([`body::perform`]), so the act's own gating (nothing carried, nobody here,
//!   a mode a device does not admit) is the route's, mapped to `409` by
//!   [`enact_response`]. It is never the character's mistake to `500`: every verb
//!   here is a body act, so none falls through to `NotOfTheBody`.
//!
//! # It does not remove the compiled acts, and that is not a dual path
//!
//! The same act reaches [`body::perform`] two ways — as a compiled body act in a
//! turn's grammar, and as a `/here` route — exactly as an `AtPart` act reaches it
//! both as a compiled act and as a [`station`](crate::effector::station) route.
//! There is one implementation (the world's own dispatch) with two front doors,
//! not two implementations, so the no-dual-path rule is kept.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde_json::{json, Map, Value};

use candle_conversation::stencil::{ParamType, ToolSpec};

use crate::api::err;
use crate::effector::auth::DeviceCaller;
use crate::effector::enact_route::enact_response;
use crate::effector::router::Local;
use crate::engine::act::Act;
use crate::engine::body;
use crate::engine::tools::{self, Mode};

/// The world-state acts `/here` mounts — the catalogue's non-part-bound acts on
/// things and rooms, less the ones another surface already owns and the ones the
/// design keeps as embodied body acts.
///
/// **Excluded on purpose.** Speech, movement, `act`, `sleep`, `promise` and
/// `remind` stay compiled body acts (the effector device is for the world's
/// affordances, not for a body's human and social interactions). The lift keeps
/// its dedicated `/lift` routes; the phone acts keep `/phone`; the tower's
/// `command_tower`/`produce` are `AtPart`, so the station mechanism already
/// mounts them where their fixtures stand.
///
/// Membership is what a `POST` is checked against; whether a listed act is
/// *available this moment* is [`specs_within`](tools::specs_within)'s call, so an
/// act in this list can still be absent from `GET`/`OPTIONS` and refused by a
/// `POST` when the room offers it nothing to work on.
const HERE_VERBS: &[&str] = &[
    "read",
    "scan",
    "claim",
    "release",
    "operate",
    "post_notice",
    "record_verdict",
    "give",
    "equip",
    "use",
    "gather",
    "engage",
    "recall",
];

/// The `/here` sub-router: the index, the schema, and one act route.
///
/// Returned without state so the outer router supplies its [`Local`] and the
/// device-auth layer, exactly as the station and `/self` routes are mounted.
pub fn routes() -> Router<Local> {
    Router::new()
        .route("/", get(index).options(schema))
        .route("/:verb", post(invoke))
}

/// `GET /here` — the acts available where the body stands.
///
/// Exactly the world-state acts [`specs_within`](tools::specs_within) admits for
/// this standpoint, each with its one-line description — so an act with nothing
/// to work on is absent rather than listed and refused.
async fn index(State(local): State<Local>, Extension(caller): Extension<DeviceCaller>) -> Response {
    let verbs: Vec<Value> = available(&local, &caller)
        .into_iter()
        .map(|spec| {
            json!({
                "name": spec.name,
                "summary": tools::by_name(&spec.name).map(summary).unwrap_or_default(),
                "choices": live_choices(&spec),
            })
        })
        .collect();
    Json(json!({
        "id": "here",
        "summary": "What you can do where you stand — with what is around you and what you carry. \
                    Each verb lists the values its arguments take right now under `choices`.",
        "verbs": verbs,
    }))
    .into_response()
}

/// `OPTIONS /here` — the schema (effector design §5.1, Appendix G).
///
/// One `POST` verb per available act, each body the JSON schema of its
/// arguments: enumerable properties carry their live enum from world state, free
/// properties are plain strings. Built from the same [`specs_within`](tools::
/// specs_within) `GET` lists, so the schema and the index never disagree.
async fn schema(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
) -> Response {
    let mut verbs = Map::new();
    for spec in available(&local, &caller) {
        verbs.insert(spec.name.clone(), json!({ "body": body_schema(&spec) }));
    }
    Json(json!({
        "id": "here",
        "summary": "What you can do where you stand — with what is around you and what you carry.",
        "methods": { "POST": { "verbs": verbs } },
    }))
    .into_response()
}

/// `POST /here/:verb` — act where the body stands.
///
/// The verb must be one `/here` owns ([`HERE_VERBS`]), or it is not something
/// that can be done here (`404`). Only the act's declared params are carried
/// through, each as the body gave it, and the act is run through the real
/// dispatch — so the act's own gating is the route's, mapped to `409` by
/// [`enact_response`]. Availability is not pre-checked here: an act the room
/// offers nothing to work on refuses itself in the world's own words, which is
/// the prescriptive error a character corrects against (§12).
async fn invoke(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(verb): Path<String>,
    payload: Option<Json<Value>>,
) -> Response {
    if !HERE_VERBS.contains(&verb.as_str()) {
        return err(
            StatusCode::NOT_FOUND,
            "unknown_verb",
            &format!("`{verb}` is not something you can do where you stand"),
        );
    }
    let Some((hosted, body)) = local.body_of(&caller) else {
        return no_body();
    };
    // A verb in `HERE_VERBS` is a catalogue act by construction — the list is a
    // subset of the catalogue's own names.
    let tool = tools::by_name(&verb).expect("a here verb is a catalogue act");
    let mut args = Map::new();
    if let Some(Json(Value::Object(fields))) = payload {
        for p in tool.params {
            if let Some(value) = fields.get(p.name) {
                args.insert(p.name.to_string(), value.clone());
            }
        }
    }
    let act = Act {
        tool: tool.name,
        args,
    };
    enact_response(body::perform(&hosted, &body, &act))
}

/// The world-state acts available for the caller's standpoint, in catalogue
/// order.
///
/// The one place the live computation is consulted: [`Runtime::within`](crate::
/// engine::runtime::Runtime::within) assembles what is true where the body
/// stands, and [`specs_within`](tools::specs_within) turns it into the acts the
/// grammar would offer with their live enums. `/here` keeps the ones it owns.
///
/// Empty when the runtime has been dropped or the caller has no body — the same
/// safe-to-read degradation the personal surfaces follow (§7.2).
fn available(local: &Local, caller: &DeviceCaller) -> Vec<ToolSpec> {
    let Some(runtime) = local.runtime() else {
        return Vec::new();
    };
    let within = runtime.within(caller.npc_id);
    tools::specs_within(Mode::Physical, &within)
        .into_iter()
        .filter(|spec| HERE_VERBS.contains(&spec.name.as_str()))
        .collect()
}

/// One act's body schema: each param a property typed by its `ty`, with its live
/// `enum` where the world enumerates it, and the required ones listed.
///
/// This is where `/here` differs from the station schema ([`crate::effector::
/// station::body_schema`], which never invents an enum): the enum here is not
/// invented, it is the live `Choices` set [`specs_within`](tools::specs_within)
/// already computed for this standpoint, so a character reading the schema is
/// told exactly which `what`/`on`/`mode` values the act will accept right now.
fn body_schema(spec: &ToolSpec) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for p in &spec.params {
        let mut prop = Map::new();
        prop.insert("type".to_string(), json!(type_name(p.ty)));
        if let Some(values) = &p.enum_values {
            prop.insert("enum".to_string(), json!(values));
        }
        properties.insert(p.name.clone(), Value::Object(prop));
        if p.required {
            required.push(Value::String(p.name.clone()));
        }
    }
    json!({ "type": "object", "properties": properties, "required": required })
}

/// The values each enumerated argument of an act may take right now, by
/// argument name — what stands here, drawn from the live `Choices` rather than
/// described in prose. Free-text arguments are absent.
fn live_choices(spec: &ToolSpec) -> Value {
    let mut choices = Map::new();
    for p in &spec.params {
        if let Some(values) = &p.enum_values {
            choices.insert(p.name.clone(), json!(values));
        }
    }
    Value::Object(choices)
}

/// A parameter type as it appears on the wire — the lowercase JSON-schema name.
/// Every world-state act's params are strings or booleans (the catalogue's hard
/// rule), but the whole enum is mapped so a future typed param renders honestly.
fn type_name(ty: ParamType) -> &'static str {
    match ty {
        ParamType::String => "string",
        ParamType::Integer => "integer",
        ParamType::Number => "number",
        ParamType::Boolean => "boolean",
        ParamType::Array => "array",
        ParamType::Object => "object",
    }
}

/// An act's one-line description, for the index — its first sentence, so the
/// listing stays scannable while `OPTIONS` carries the full schema.
fn summary(tool: &tools::Tool) -> String {
    match tool.description.split_once(". ") {
        Some((first, _)) => format!("{first}."),
        None => tool.description.to_string(),
    }
}

/// The refusal for a caller whose body cannot be resolved to a hosted world —
/// there is nothing to do where a body that is nowhere stands.
fn no_body() -> Response {
    err(
        StatusCode::NOT_FOUND,
        "no_body",
        "you have no body in a running world, so there is nothing here to do",
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

    use super::HERE_VERBS;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
    const WORLD: &str = "creators-vault";

    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-here-ut-{}-{}",
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
                Where::new("vault-chronicle", "early-range"),
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

    fn verb_names(index: &Value) -> Vec<String> {
        index["verbs"]
            .as_array()
            .expect("a verbs array")
            .iter()
            .map(|v| v["name"].as_str().unwrap().to_string())
            .collect()
    }

    /// **`GET /here` lists the always-available world-state acts.**
    /// `record_verdict`'s targets are free text, so it is reachable in any room
    /// a body stands in — the floor the index must always show. `release` is not
    /// on that floor: it is listed only once something is held.
    #[tokio::test]
    async fn the_index_lists_the_always_available_acts() {
        let (rt, tokens) = daemon(60);
        let (status, index) = send(&rt, &tokens, 60, "GET", "/here", None).await;
        assert_eq!(status, StatusCode::OK, "{index}");
        let names = verb_names(&index);
        assert!(
            !names.contains(&"release".to_string()),
            "release was listed to a body holding nothing: {index}"
        );
        assert!(
            names.contains(&"record_verdict".to_string()),
            "record_verdict is always available: {index}"
        );
        // Every listed verb is one `/here` owns — the index never advertises an
        // act mounted on another surface.
        for name in &names {
            assert!(
                HERE_VERBS.contains(&name.as_str()),
                "`{name}` is not a /here verb: {index}"
            );
        }
    }

    /// **`OPTIONS /here` carries a body schema per available verb**, agreeing
    /// with the index on the verb set. `record_verdict`'s schema names its two
    /// required fields and its optional one — the raw shape a `POST` accepts.
    #[tokio::test]
    async fn options_carries_body_schemas() {
        let (rt, tokens) = daemon(61);
        let (status, index) = send(&rt, &tokens, 61, "GET", "/here", None).await;
        assert_eq!(status, StatusCode::OK, "{index}");
        let listed = verb_names(&index);

        let (status, schema) = send(&rt, &tokens, 61, "OPTIONS", "/here", None).await;
        assert_eq!(status, StatusCode::OK, "{schema}");
        let verbs = schema["methods"]["POST"]["verbs"]
            .as_object()
            .expect("a verbs object");
        for name in &listed {
            assert!(
                verbs.contains_key(name),
                "`{name}` was listed but not schema'd: {schema}"
            );
        }
        assert_eq!(
            schema["methods"]["POST"]["verbs"]["record_verdict"]["body"],
            json!({
                "type": "object",
                "properties": {
                    "on": { "type": "string" },
                    "judgement": { "type": "string" },
                    "what_would_change_it": { "type": "string" },
                },
                "required": ["on", "judgement"]
            }),
            "{schema}"
        );
    }

    /// **A listed enumerable argument carries its live enum.** `scan`'s `at` is
    /// bound to the places this body can look at; where the room has neighbours
    /// the schema advertises them as an `enum`, which is the whole point of the
    /// surface — a character is told the valid targets, not left to guess.
    #[tokio::test]
    async fn an_enumerable_argument_carries_its_live_enum() {
        let (rt, tokens) = daemon(62);
        let (status, schema) = send(&rt, &tokens, 62, "OPTIONS", "/here", None).await;
        assert_eq!(status, StatusCode::OK, "{schema}");
        if let Some(scan) = schema["methods"]["POST"]["verbs"].get("scan") {
            let at = &scan["body"]["properties"]["at"];
            assert_eq!(at["type"], json!("string"), "{schema}");
            assert!(
                at["enum"].as_array().is_some_and(|e| !e.is_empty()),
                "scan's `at` did not carry its live places: {schema}"
            );
        }
    }

    /// **Every listed verb runs the real act and never `500`s.** Whether the act
    /// does its thing (`200`) or refuses (`409`) is the world's call; what
    /// matters is that none falls through to `NotOfTheBody`, because every act
    /// `/here` mounts is a body act.
    #[tokio::test]
    async fn every_listed_verb_runs_the_real_act() {
        let (rt, tokens) = daemon(63);
        let (_, index) = send(&rt, &tokens, 63, "GET", "/here", None).await;
        for name in verb_names(&index) {
            let (status, out) = send(
                &rt,
                &tokens,
                63,
                "POST",
                &format!("/here/{name}"),
                Some(json!({})),
            )
            .await;
            assert!(
                status == StatusCode::OK || status == StatusCode::CONFLICT,
                "`{name}` did not run the real act (status {status}): {out}"
            );
            assert_ne!(
                status,
                StatusCode::INTERNAL_SERVER_ERROR,
                "`{name}` fell through to NotOfTheBody: {out}"
            );
        }
    }

    /// **`release` appears in the index once something is held, and goes again
    /// when it is given back.**
    #[tokio::test]
    async fn release_is_listed_only_while_something_is_held() {
        let (rt, tokens) = daemon(65);
        let order = "close the longest silence in the record";
        let (status, out) = send(
            &rt,
            &tokens,
            65,
            "POST",
            "/here/claim",
            Some(json!({ "what": order })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{out}");

        let (_, index) = send(&rt, &tokens, 65, "GET", "/here", None).await;
        assert!(
            verb_names(&index).contains(&"release".to_string()),
            "release was not listed while an order was held: {index}"
        );

        let (status, out) = send(&rt, &tokens, 65, "POST", "/here/release", Some(json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{out}");
        let (_, index) = send(&rt, &tokens, 65, "GET", "/here", None).await;
        assert!(
            !verb_names(&index).contains(&"release".to_string()),
            "release was still listed after giving back: {index}"
        );
    }

    /// **An unknown verb is a `404`, not a `500`.** It is not a verb `/here`
    /// owns, so it is refused before any act is synthesised.
    #[tokio::test]
    async fn an_unknown_verb_is_not_found() {
        let (rt, tokens) = daemon(64);
        let (status, out) = send(
            &rt,
            &tokens,
            64,
            "POST",
            "/here/frobnicate",
            Some(json!({})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{out}");
        assert_eq!(out["error"], json!("unknown_verb"), "{out}");
    }
}
