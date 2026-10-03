//! Every station namespace's verbs, lifted onto `http://local` at once.
//!
//! [`lift`](crate::effector::lift) hand-wrote one namespace's routes. This is
//! the generalisation the lift's shape was a template for: a single sub-router,
//! mounted once per part namespace ([`crate::effector::router`]), that turns
//! *any* placed station into a readable, schema-describing, invokable resource —
//! driven entirely by the [`Tool`] catalogue, so a new station act is a route
//! the moment its `Tool` names the part it attaches to, with no code here to
//! change (effector design §9, §12, Appendix C).
//!
//! Three routes, mounted under a namespace prefix (`/chronicle`, `/record`, …):
//!
//! - `GET /<ns>/:id` — the resource as a character reads it: the part it is, its
//!   name, the short verbs it affords here, and its live `state` — the machine's
//!   mode and, where the acts it affords draw on the tower, the stockpile, what
//!   can be made and which queues are free (`sim::reading`). `:id` is the map's own instance
//!   id (`<area>~<node>~<part-id>~<ordinal>`, [`npc_map::instance`]); it is
//!   resolved against the map and reach-checked, so an id that names nothing, or
//!   a thing not within the caller body's standpoint, is a `404` — the same
//!   answer the world gives a body reaching for what is not there.
//! - `OPTIONS /<ns>/:id` — the schema (§5.1): for the resolved part, the acts in
//!   the catalogue whose [`Tool::at`] names it, each as `POST` verb → body
//!   schema built from the act's own `params`. The schema is a pure function of
//!   the catalogue and the resolved part, so it never drifts from what a `POST`
//!   will accept.
//! - `POST /<ns>/:id/:verb` — act. The verb is resolved back to the catalogue
//!   act, the JSON body read into the act's `args`, and the act run through the
//!   real dispatch ([`body::perform`]) — so proximity, custody and mode are
//!   enforced by the act's own handler and mapped to `409` by
//!   [`enact_response`], exactly as the lift routes are (§10, §11).
//!
//! # Reach is the gate, and it is the map's answer
//!
//! An **as-npc** caller reaches only what is within its standpoint: the resolved
//! instance's place must be where the body stands (`within_reach`,
//! `perceive.rs`), or the route `404`s. A **direct**-scope caller
//! ([`DeviceCaller::scope`]) addresses any id regardless of standpoint, per
//! §7/§8.3 — which is what lets an operator drive a station by id from outside.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde_json::{json, Map, Value};

use npc_map::world::Where;

use crate::api::err;
use crate::effector::auth::{DeviceCaller, PinnedStandpoint};
use crate::effector::enact_route::enact_response;
use crate::effector::namespace::namespace_of;
use crate::effector::router::Local;
use crate::effector::token::Scope;
use crate::engine::act::Act;
use crate::engine::body;
use crate::engine::tools::{self, Tool};
use crate::world::Hosted;

/// The sub-router for one station namespace: read, schema, act.
///
/// The *same* router is nested under every station prefix
/// ([`crate::effector::router::router`]); what tells one mount from another is
/// the [`StationNamespace`] extension the caller layers onto each nest, which
/// every handler here checks the resolved part's own namespace
/// ([`namespace_of`]) against — so `/portrait/<id>` and `/chronicle/<id>`
/// cannot silently answer for each other's parts (§9, "each namespace is its
/// own static prefix"). Returned without state applied so the outer router
/// supplies its [`Local`] and its device-auth layer to these routes too.
pub fn routes() -> Router<Local> {
    Router::new()
        .route("/:id", get(state).options(schema))
        .route("/:id/:verb", post(invoke))
}

/// The namespace prefix a station nest was mounted under
/// ([`crate::effector::router::router`]), carried as a request extension so a
/// handler can tell which of the several identical `station::routes()` mounts
/// answered this call.
#[derive(Clone, Copy)]
pub struct StationNamespace(pub &'static str);

/// `GET /<ns>/:id` — the resource as the character reads it.
///
/// The descriptor a character acts against: which part this is, what it is
/// called, and the short verbs it affords — the same verbs `OPTIONS` describes
/// and `POST` accepts, so what the read advertises is exactly what can be done.
async fn state(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Extension(mount): Extension<StationNamespace>,
    pinned: Option<Extension<PinnedStandpoint>>,
    Path(id): Path<String>,
) -> Response {
    let Some((hosted, _body, part)) = resolve(&local, &caller, pinned_at(pinned), &id) else {
        return not_here(&id);
    };
    let ns = namespace_of(&part.part_id);
    if mount.0 != ns {
        return not_here(&id);
    }
    let verbs: Vec<String> = acts_at(&part.part_id)
        .map(|t| short_verb(t.name, ns).to_string())
        .collect();
    let acts: Vec<&str> = acts_at(&part.part_id).map(|t| t.name).collect();
    let state = hosted.with_both(|world, sim| {
        let who = |body: &str| {
            world
                .actor(body)
                .map_or_else(|| body.to_string(), |a| a.name.clone())
        };
        sim.reading(&id, &part.at, &acts, &who)
    });
    let invoke: Vec<String> = verbs
        .iter()
        .map(|v| format!("http://local/{ns}/{id}/{v}"))
        .collect();
    Json(json!({
        "id": format!("{ns}/{id}"),
        "part": part.part_id,
        "name": part.name,
        "verbs": verbs,
        "invoke": invoke,
        "state": state,
    }))
    .into_response()
}

/// `OPTIONS /<ns>/:id` — the resource schema (effector design §5.1).
///
/// One `POST` verb per act the catalogue attaches to this part, each carrying
/// the JSON schema of its body built from the act's own `params`: every param a
/// property typed by its `ty`, the required ones listed. The types are passed
/// through as the catalogue declares them (`string`/`boolean` today) — no enum
/// is invented here; the handler validates the rest (§11/§12).
async fn schema(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Extension(mount): Extension<StationNamespace>,
    pinned: Option<Extension<PinnedStandpoint>>,
    Path(id): Path<String>,
) -> Response {
    let Some((_hosted, _body, part)) = resolve(&local, &caller, pinned_at(pinned), &id) else {
        return not_here(&id);
    };
    let ns = namespace_of(&part.part_id);
    if mount.0 != ns {
        return not_here(&id);
    }
    let mut verbs = Map::new();
    for t in acts_at(&part.part_id) {
        verbs.insert(
            short_verb(t.name, ns).to_string(),
            json!({ "body": body_schema(t) }),
        );
    }
    Json(json!({
        "id": format!("{ns}/{id}"),
        "summary": part.summary,
        "methods": { "POST": { "verbs": verbs } },
    }))
    .into_response()
}

/// `POST /<ns>/:id/:verb` — act at the resource.
///
/// The verb is resolved back to the catalogue act — `<ns>_<verb>` for the common
/// case an act is named for its own namespace, with the act's own name as the
/// fallback for the few that are not (the craft libraries, `library_*`, reach
/// the character and story surfaces under their own name). The act must attach
/// to this part or the verb is unknown here (`404`). The body is read into the
/// act's `args` and run through the real dispatch, so the world's own rules — and
/// its own words on refusing — are the route's ([`enact_response`]).
async fn invoke(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Extension(mount): Extension<StationNamespace>,
    pinned: Option<Extension<PinnedStandpoint>>,
    Path((id, verb)): Path<(String, String)>,
    payload: Option<Json<Value>>,
) -> Response {
    let Some((hosted, body, part)) = resolve(&local, &caller, pinned_at(pinned), &id) else {
        return not_here(&id);
    };
    let ns = namespace_of(&part.part_id);
    if mount.0 != ns {
        return not_here(&id);
    }
    // `<ns>_<verb>` reconstructs the act name for the common case; the bare verb
    // is the fallback for an act whose name is not its namespace's (`library_*`).
    // Either way it must be an act that attaches to *this* part, or it is not a
    // verb this resource affords.
    let reconstructed = format!("{ns}_{verb}");
    let Some(tool) = [reconstructed.as_str(), verb.as_str()]
        .into_iter()
        .filter_map(tools::by_name)
        .find(|t| t.at.contains(&part.part_id.as_str()))
    else {
        return err(
            StatusCode::NOT_FOUND,
            "unknown_verb",
            &format!("`{verb}` is not something you can do at {}", part.name),
        );
    };
    // Only the act's declared params are carried through, each as the body gave
    // it — the handler reads strings and booleans off the map exactly as it does
    // when the act is reached any other way.
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

/// What a resolved station is, in the fields the routes answer from — owned, so
/// the world lock is released before the response is built or an act is run.
struct Station {
    /// The catalogue part id — `chronicle-terminal`, `accession-desk`.
    part_id: String,
    /// The `area/node` room it stands in, the key the sim answers by.
    at: String,
    /// What the part is called, singular and bare.
    name: String,
    /// What it is and does, for the schema summary ([`Part::long`], the short
    /// level-prose clause as a fallback).
    summary: String,
}

/// Resolve `:id` to the station it names, reach-checked against the caller.
///
/// The one place the map is consulted: it splits the instance id back to its
/// place and part ([`MapSet::resolve_instance`]), reads the caller body's
/// standpoint, and — for an as-npc caller — refuses anything not standing where
/// the body stands. `None` when the caller has no body, the id resolves to
/// nothing, or the thing is out of reach — the three states a route answers
/// alike, as [`not_here`]. `Some` carries the world handle and body id the act
/// path needs, with the world lock already released.
fn resolve(
    local: &Local,
    caller: &DeviceCaller,
    pinned: Option<Where>,
    id: &str,
) -> Option<(Arc<Hosted>, String, Station)> {
    let (hosted, body) = local.body_of(caller)?;
    let (station, place, live_standpoint) = hosted.read(|w| {
        let inst = w.map().resolve_instance(id)?;
        let part = inst.part();
        let summary = match part.long.trim().is_empty() {
            false => part.long.clone(),
            true => part.short.clone().unwrap_or_else(|| part.name.clone()),
        };
        let place = inst.place().clone();
        let station = Station {
            part_id: inst.part_id().to_string(),
            at: format!("{}/{}", place.area, place.node),
            name: inst.name().to_string(),
            summary,
        };
        let standpoint = w.actor(&body).map(|a| a.at.clone());
        Some((station, place, standpoint))
    })?;
    // Reach is checked against the *pinned* standpoint when the in-process fast
    // path supplied one — the position the body stood in when this turn's
    // grammar offered the address — falling back to the live position for an
    // external caller (which carries no pin). This closes the time-of-check /
    // time-of-use gap where a body walks a leg during its own decode and the
    // live check then 404s an address the grammar had just approved
    // ([`PinnedStandpoint`]).
    let standpoint = pinned.or(live_standpoint);
    // As-npc is proximity-gated (§8.3): the thing must be where the body stands.
    // A direct-scope caller addresses any id regardless of standpoint.
    if caller.scope != Scope::Direct && standpoint.as_ref() != Some(&place) {
        return None;
    }
    Some((hosted, body, station))
}

/// The `Where` a request pins its reach check to, if the in-process fast path
/// supplied one. Unwraps the optional [`PinnedStandpoint`] extension; an
/// external request never carries it, so the reach check uses the live position.
fn pinned_at(pinned: Option<Extension<PinnedStandpoint>>) -> Option<Where> {
    pinned.map(|Extension(PinnedStandpoint(at))| at)
}

/// The catalogue acts that attach to a part, in catalogue order.
fn acts_at(part_id: &str) -> impl Iterator<Item = &'static Tool> + '_ {
    tools::CATALOG
        .iter()
        .filter(move |t| t.at.contains(&part_id))
}

/// The verbs a part affords at its namespace, each with the act behind it — the
/// `<verb>` half of a `POST /<ns>/<id>/<verb>`, in catalogue order.
///
/// The one place the invoke verb-path is derived, so the grammar's `invoke` url
/// enum ([`tools::Choices::InvokeUrl`], built in [`crate::engine::runtime::
/// Runtime`]) and the route that answers the call name the same verb. The act
/// rides along because it is what the body is typed by: the verb's fields are
/// the act's own parameters. A part that affords nothing (a seat) yields
/// nothing, so it contributes no invoke address — you cannot act on a chair.
pub fn verbs_at(part_id: &str) -> Vec<(String, &'static Tool)> {
    let ns = namespace_of(part_id);
    acts_at(part_id)
        .map(|t| (short_verb(t.name, ns).to_string(), t))
        .collect()
}

/// An act's short verb under a namespace: its name with the `<ns>_` prefix
/// stripped, or the whole name when it does not carry that prefix.
fn short_verb<'a>(name: &'a str, ns: &str) -> &'a str {
    name.strip_prefix(&format!("{ns}_")).unwrap_or(name)
}

/// The JSON schema of an act's body: each param a property typed by its `ty`,
/// the required ones listed. Types are passed through, never invented (§11).
///
/// `pub(crate)` so the personal `/phone` routes build their verb schemas the one
/// way the station routes do — the same raw shape, from each act's own `params`.
pub(crate) fn body_schema(tool: &Tool) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for p in tool.params {
        properties.insert(p.name.to_string(), json!({ "type": p.ty }));
        if p.required {
            required.push(Value::String(p.name.to_string()));
        }
    }
    json!({ "type": "object", "properties": properties, "required": required })
}

/// The `404` for an id that resolves to nothing, or to something out of reach —
/// one answer for "no such thing here", the map's own verdict on where a body is.
fn not_here(id: &str) -> Response {
    err(
        StatusCode::NOT_FOUND,
        "not_found",
        &format!("there is no `{id}` within your reach"),
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

    use crate::effector::auth::PinnedStandpoint;
    use crate::effector::router::{router, Local};
    use crate::effector::token::{Scope, Tokens};
    use crate::engine::runtime::Runtime;
    use crate::mind::Mind;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
    const WORLD: &str = "creators-vault";

    /// The instance id of a real chronicle terminal in the shipped vault, derived
    /// from the map rather than hard-coded — the id is `<part>~<ordinal>` and the
    /// ordinal is the map's to decide (see [`npc_map::instance`]).
    fn chronicle_terminal() -> String {
        use npc_map::load::MapSet;
        use npc_map::schema::Where;
        MapSet::load_dir(ROOMS)
            .expect("the vault loads")
            .instances_at(&Where::new("vault-chronicle", "early-range"))
            .into_iter()
            .find(|i| i.part_id() == "chronicle-terminal")
            .expect("a chronicle terminal stands in the early range")
            .id()
    }

    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-station-ut-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// A held-still vault with the device installed and a body standing where the
    /// chronicle terminal is, so it is within reach.
    fn daemon(npc_id: u64) -> (Arc<Runtime>, Arc<Tokens>) {
        daemon_at(WORLD, Where::new("vault-chronicle", "early-range"), npc_id)
    }

    /// [`daemon`] for any world, with the body standing at `at`.
    fn daemon_at(world_id: &str, at: Where, npc_id: u64) -> (Arc<Runtime>, Arc<Tokens>) {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
        rt.host(world_id, Path::new(ROOMS))
            .expect("the world loads");
        rt.hold_world(world_id, true);
        let tokens = Arc::new(Tokens::load(tmp()).expect("a fresh token store"));
        rt.set_tokens(tokens.clone());

        let body = Runtime::body_id(npc_id);
        let world = rt.hosted.get(world_id).expect("hosted");
        world.with(|w| {
            w.enter(&body, format!("Maker-{npc_id:02}"), at)
                .expect("a real room");
        });
        rt.bodies.bind(npc_id, world_id, &body).expect("bound");
        (rt, tokens)
    }

    /// **A fabricator's read answers what a character was asking its
    /// neighbours.** The mode it is in and the stockpile behind it come back in
    /// `state`, so "which of them run, and what is there to make from?" has an
    /// answer at the machine.
    #[tokio::test]
    async fn a_fabricator_reads_the_stockpile_it_draws_on() {
        use npc_map::load::MapSet;
        let fabricator = MapSet::load_dir(ROOMS)
            .expect("the maps load")
            .instances_at(&Where::new("tower-redoubt", "foundry"))
            .into_iter()
            .find(|i| i.part_id() == "fabricator")
            .expect("a fabricator stands in the foundry")
            .id();
        let (rt, tokens) = daemon_at("battle-cities", Where::new("tower-redoubt", "foundry"), 52);

        let (status, read) = send(
            &rt,
            &tokens,
            52,
            "GET",
            &format!("/stores/{fabricator}"),
            None,
        )
        .await;

        assert_eq!(status, StatusCode::OK, "{read}");
        assert_eq!(read["state"]["mode"], json!("idle"), "{read}");
        assert_eq!(read["state"]["stockpile"]["nanobots"], json!(12), "{read}");
        assert!(
            read["state"]["can_make"]
                .as_array()
                .is_some_and(|m| m.contains(&json!("bolt rounds"))),
            "{read}"
        );
        assert_eq!(
            read["state"]["free_queues"].as_array().map(Vec::len),
            Some(8),
            "{read}"
        );
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

    /// [`send`], but stamping the request with a [`PinnedStandpoint`] the way the
    /// in-process fast path does — so the reach check runs against `pinned`, not
    /// the body's live position.
    async fn send_pinned(
        rt: &Arc<Runtime>,
        tokens: &Arc<Tokens>,
        npc_id: u64,
        method: &str,
        path: &str,
        pinned: Where,
    ) -> (StatusCode, Value) {
        let token = tokens.mint(npc_id, Scope::AsNpc).expect("minted");
        let app = router(Local::new(tokens.clone(), rt));
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .extension(PinnedStandpoint(pinned))
            .body(Body::empty())
            .unwrap();
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

    /// **A station answers only under its own namespace prefix.** A chronicle
    /// terminal's id resolves fine under `/chronicle`, but the identical id
    /// requested under an unrelated mounted namespace (`/portrait`) must not
    /// silently answer for it — each namespace is its own static prefix (§9),
    /// not a detail `resolve` ignores.
    #[tokio::test]
    async fn a_station_does_not_answer_under_a_different_namespace() {
        let (rt, tokens) = daemon(51);
        let terminal = chronicle_terminal();
        let wrong_path = format!("/portrait/{terminal}");

        let (status, body) = send(&rt, &tokens, 51, "GET", &wrong_path, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");

        let (status, body) = send(&rt, &tokens, 51, "OPTIONS", &wrong_path, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");

        let (status, body) = send(
            &rt,
            &tokens,
            51,
            "POST",
            &format!("{wrong_path}/add_entry"),
            Some(json!({ "to": "x", "what": "y" })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");

        // And its own namespace still answers, unaffected.
        let (status, body) = send(
            &rt,
            &tokens,
            51,
            "GET",
            &format!("/chronicle/{terminal}"),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    /// **The read/schema/invoke round-trip on one station instance.** `GET` names
    /// the part and the verbs it affords; `OPTIONS` describes the same verbs with
    /// their raw body shapes; `POST` runs the real act and hands back the world's
    /// own verdict — here a `409` refusal, because the seeded "third era" is filed.
    /// The three answers agree on the verb set, which is the whole point: what the
    /// read advertises is exactly what a `POST` accepts.
    #[tokio::test]
    async fn a_station_reads_schemas_and_invokes_consistently() {
        let (rt, tokens) = daemon(50);
        let terminal = chronicle_terminal();
        let path = format!("/chronicle/{terminal}");

        // GET — the descriptor.
        let (status, read) = send(&rt, &tokens, 50, "GET", &path, None).await;
        assert_eq!(status, StatusCode::OK, "{read}");
        assert_eq!(read["id"], json!(format!("chronicle/{terminal}")));
        assert_eq!(read["part"], json!("chronicle-terminal"));
        let read_verbs: Vec<String> = read["verbs"]
            .as_array()
            .expect("a verbs array")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(
            read_verbs.contains(&"add_entry".to_string()),
            "the descriptor did not name add_entry: {read}"
        );
        assert!(
            read["invoke"]
                .as_array()
                .expect("an invoke array")
                .contains(&json!(format!(
                    "http://local/chronicle/{terminal}/add_entry"
                ))),
            "the descriptor did not give the full invoke address: {read}"
        );

        // OPTIONS — the schema over the same verb set.
        let (status, schema) = send(&rt, &tokens, 50, "OPTIONS", &path, None).await;
        assert_eq!(status, StatusCode::OK, "{schema}");
        let schema_verbs = schema["methods"]["POST"]["verbs"]
            .as_object()
            .expect("a verbs object");
        for v in &read_verbs {
            assert!(
                schema_verbs.contains_key(v),
                "`{v}` was read but not in the schema: {schema}"
            );
        }
        assert_eq!(
            schema["methods"]["POST"]["verbs"]["add_entry"]["body"],
            json!({
                "type": "object",
                "properties": { "to": { "type": "string" }, "what": { "type": "string" } },
                "required": ["to", "what"]
            }),
            "{schema}"
        );

        // POST — the real act runs, and its refusal (the third era is filed)
        // comes back as a 409 in the world's own words.
        let (status, refused) = send(
            &rt,
            &tokens,
            50,
            "POST",
            &format!("{path}/add_entry"),
            Some(json!({ "to": "the third era", "what": "and the year it was not rebuilt" })),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{refused}");
        assert_eq!(refused["error"], json!("refused"), "{refused}");
        assert!(
            refused["detail"]
                .as_str()
                .is_some_and(|d| d.contains("is filed")),
            "the handler's own refusal was not carried through: {refused}"
        );
    }

    /// **An unknown verb at a real station is a `404`, not a `500`.** It
    /// reconstructs to no catalogue act that attaches here, so the resource does
    /// not afford it — refused before any act is synthesised.
    #[tokio::test]
    async fn an_unknown_verb_is_not_found() {
        let (rt, tokens) = daemon(51);
        let terminal = chronicle_terminal();
        let (status, out) = send(
            &rt,
            &tokens,
            51,
            "POST",
            &format!("/chronicle/{terminal}/frobnicate"),
            Some(json!({ "to": "the third era", "what": "anything" })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{out}");
        assert_eq!(out["error"], json!("unknown_verb"), "{out}");
    }

    /// **A pinned standpoint reaches a station the body has since walked away
    /// from.** This is the walk-while-you-think fix ([`PinnedStandpoint`]): a
    /// character's grammar is snapshotted before its decode, but a moving body
    /// covers a leg every metronome tick during the seconds the decode takes, so
    /// the chosen `invoke` can land after the body has left the room the
    /// address named. The in-process fast path pins the grammar-time standpoint,
    /// and the reach check honours it — while an external caller, which carries no
    /// pin, is still gated by its live position.
    #[tokio::test]
    async fn a_pinned_standpoint_reaches_a_station_the_body_has_left() {
        let (rt, tokens) = daemon(60);
        let terminal = chronicle_terminal();
        let path = format!("/chronicle/{terminal}");
        // The terminal stands in the early range; the body has walked out to a
        // corridor that holds nothing, so it is no longer within live reach.
        let body = Runtime::body_id(60);
        let world = rt.hosted.get(WORLD).expect("hosted");
        world.with(|w| {
            w.place(&body, Where::new("vault-casting", "ring-north"))
                .expect("a real corridor");
        });

        // Unpinned — the live reach check refuses it, because the body is gone.
        let (status, _) = send(&rt, &tokens, 60, "GET", &path, None).await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "the live reach should refuse a station the body has left"
        );

        // Pinned to where the body stood when the grammar was built — it resolves,
        // exactly as the address the grammar offered that turn should.
        let (status, read) = send_pinned(
            &rt,
            &tokens,
            60,
            "GET",
            &path,
            Where::new("vault-chronicle", "early-range"),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "the pinned reach should resolve the grammar-time station: {read}"
        );
        assert_eq!(read["part"], json!("chronicle-terminal"), "{read}");
    }
}
