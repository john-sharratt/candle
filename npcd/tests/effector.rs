//! The effector device, stood up against the real vault and driven the two ways
//! it will be driven for real: over a socket (via `oneshot`) and through the
//! engine's in-process fast path.
//!
//! This is the assembly test for the foundation slice. [`effector::token`] and
//! [`effector::auth`] have their own unit tests; this proves the pieces wired
//! together answer `GET http://local/` from where a body actually stands in a
//! hosted world — the near-you index as a pure function of world state (effector
//! design §6, §13) — and that the device surface refuses the gateway's identity
//! headers exactly as it refuses no credential at all.
//!
//! The world is held still throughout: these assert what standing somewhere
//! *makes reachable*, and a metronome beating underneath would make each a race.

use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use npc_map::world::Where;

use npcd::effector::router::{router, Local};
use npcd::effector::token::{Scope, Tokens};
use npcd::engine::runtime::Runtime;
use npcd::mind::Mind;

const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
const WORLD: &str = "creators-vault";

/// A directory of this test's own, counted so two tests never share one.
fn tmp() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let p = std::env::temp_dir().join(format!(
        "npcd-effector-it-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// A daemon hosting the vault, held still, with the effector device installed —
/// the token store and the `local` router the fast path drives.
fn daemon() -> (Arc<Runtime>, Arc<Tokens>) {
    let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
    rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
    rt.hold_world(WORLD, true);

    let tokens = Arc::new(Tokens::load(tmp()).expect("a fresh token store"));
    rt.set_tokens(tokens.clone());
    rt.set_effector_router(router(Local::new(tokens.clone(), &rt)));
    (rt, tokens)
}

/// **Reproduction: the effector nest survives being added to a merged, stated
/// parent — the shape `main.rs` builds.** The isolated-nest test nests into a
/// bare `Router::new()`; this nests into a parent that already carries sibling
/// `/v1` routes merged from several routers, which is how the daemon assembles
/// it. Probed without a token: a route that exists answers `401` (device auth), a
/// route the nest failed to register answers `404`.
#[tokio::test]
async fn the_effector_nest_survives_a_merged_parent() {
    use axum::routing::get;
    let (rt, _tokens) = daemon();
    let effector = router(Local::new(rt.tokens().expect("token store"), &rt));
    let parent: axum::Router = axum::Router::new()
        .route("/v1/world", get(|| async { "ok" }))
        // A merged router that carries a fallback — the shape `engine::api`'s
        // "awaiting an engine" routes give the assembled router in `main.rs`.
        .merge(
            axum::Router::new()
                .route("/v1/thing", get(|| async { "ok" }))
                .fallback(|| async { (StatusCode::NOT_FOUND, "no engine") }),
        )
        .nest("/v1/local", effector);
    for path in ["/v1/local", "/v1/local/phone", "/v1/local/here"] {
        let res = parent
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::UNAUTHORIZED,
            "`{path}` was not reachable through the merged nest (got {})",
            res.status()
        );
    }
}

/// Put a character's body into the vault at a standpoint, and bind it, so the
/// near-you index can resolve it from the npc id.
fn place(rt: &Arc<Runtime>, npc_id: u64, node: &str) {
    let body = Runtime::body_id(npc_id);
    let world = rt.hosted.get(WORLD).expect("hosted");
    world.with(|w| {
        w.enter(
            &body,
            format!("Maker-{npc_id:02}"),
            Where::new("vault-casting", node),
        )
        .expect("a real room")
    });
    rt.bodies.bind(npc_id, WORLD, &body).expect("bound");
}

/// Put a character's body at a specific standpoint (a lift landing is a level's
/// `core` node, reached through the shaft rather than by room name), and bind it.
fn place_where(rt: &Arc<Runtime>, npc_id: u64, at: Where) {
    let body = Runtime::body_id(npc_id);
    let world = rt.hosted.get(WORLD).expect("hosted");
    world.with(|w| {
        w.enter(&body, format!("Maker-{npc_id:02}"), at)
            .expect("a real standpoint")
    });
    rt.bodies.bind(npc_id, WORLD, &body).expect("bound");
}

/// Drive the assembled `local` router over the socket with any method and an
/// optional JSON body, as `(status, body)` — what an external client sends, and
/// the one way to reach `GET` and `OPTIONS`, which the fast path's `invoke` does
/// not spell.
async fn request(
    rt: &Arc<Runtime>,
    tokens: &Arc<Tokens>,
    npc_id: u64,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let token = tokens.ensure(npc_id, Scope::AsNpc).expect("a token");
    let router = router(Local::new(rt.tokens().expect("token store"), rt));
    let builder = Request::builder()
        .method(method)
        .uri(path)
        .header(AUTHORIZATION, format!("Bearer {token}"));
    let req = match body {
        Some(value) => builder
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&value).unwrap()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let res = router.oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// The near-you index for a character, over the socket, as `(status, body)`.
async fn near_you(rt: &Arc<Runtime>, tokens: &Arc<Tokens>, npc_id: u64) -> (StatusCode, Value) {
    let token = tokens.mint(npc_id, Scope::AsNpc).expect("minted");
    let router = rt
        .tokens()
        .map(|t| router(Local::new(t, rt)))
        .expect("the token store is installed");
    let req = Request::builder()
        .uri("/")
        .header(AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap();
    let res = router.oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// The urls of an index answer, in order.
fn urls(body: &Value) -> Vec<String> {
    body["routes"]
        .as_array()
        .expect("routes is an array")
        .iter()
        .map(|r| r["url"].as_str().expect("a url").to_string())
        .collect()
}

/// A body at a workstation sees its personal routes and **every** terminal
/// within reach as its own URL — and, because it is not on a landing, no lift.
#[tokio::test]
async fn a_body_at_a_workstation_sees_the_terminals_within_reach() {
    let (rt, tokens) = daemon();
    // Band one places six character terminals (`vault-casting.yaml`); it is a
    // work node, not a lift core, so no landing. Each terminal is its own
    // instance, so six distinct URLs come back under the `character` namespace
    // — the collapse-to-one of the pre-instance router is gone.
    place(&rt, 1, "band-one");

    let (status, body) = near_you(&rt, &tokens, 1).await;
    assert_eq!(status, StatusCode::OK);
    // The six terminals' urls, derived from the map so the test does not spell the
    // ordinals the map decides — each is `.../character/<part>~<n>`.
    let terminals: Vec<String> = npc_map::load::MapSet::load_dir(ROOMS)
        .expect("the vault loads")
        .instances_at(&Where::new("vault-casting", "band-one"))
        .into_iter()
        .filter(|i| i.part_id() == "character-terminal")
        .map(|i| format!("http://local/character/{}", i.id()))
        .collect();
    let mut expected = vec![
        "http://local/here".to_string(),
        "http://local/history".to_string(),
        "http://local/phone".to_string(),
        "http://local/self".to_string(),
    ];
    expected.extend(terminals);
    assert_eq!(urls(&body), expected);
}

/// A body standing on the lift's landing is offered the lift, and — because the
/// core node places no parts — nothing else beyond its personal routes.
#[tokio::test]
async fn a_body_at_a_landing_is_offered_the_lift() {
    let (rt, tokens) = daemon();
    place(&rt, 2, "core");

    let (status, body) = near_you(&rt, &tokens, 2).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        urls(&body),
        [
            "http://local/here",
            "http://local/history",
            "http://local/phone",
            "http://local/self",
            "http://local/lift/command-shaft",
        ]
    );
}

/// **The device surface never honours the gateway's headers.** A request
/// carrying only `x-tokera-*` admin identity and no bearer token is refused,
/// through the assembled `local` router — the both-directions pin the build
/// order calls for (§8.3), here against the real router rather than a stub.
#[tokio::test]
async fn the_device_surface_refuses_gateway_headers() {
    let (rt, _tokens) = daemon();
    place(&rt, 4, "band-one");

    let router = router(Local::new(rt.tokens().unwrap(), &rt));
    let req = Request::builder()
        .uri("/")
        .header("x-tokera-user", "boss")
        .header("x-tokera-provider", "google")
        .header("x-tokera-email", "johnathan.sharratt@gmail.com")
        .body(Body::empty())
        .unwrap();
    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

/// **The external mount answers through the real nest, and keeps its auth.**
/// `main.rs` nests the `local` router at `/v1/local` on the npcd router (§8.2);
/// this drives that nested path itself, not the bare router: a valid token
/// reaches the near-you index at `/v1/local/`, and the gateway's admin headers
/// alone are refused there too — so the device auth holds across the nest, which
/// is where an external client actually meets it.
#[tokio::test]
async fn the_external_mount_answers_through_the_nest_and_keeps_its_auth() {
    let (rt, tokens) = daemon();
    place(&rt, 9, "band-one");
    let token = tokens.ensure(9, Scope::AsNpc).expect("a token");

    let mounted =
        || axum::Router::new().nest("/v1/local", router(Local::new(rt.tokens().unwrap(), &rt)));

    // A valid token reaches the near-you index at the nested path. axum serves a
    // nested router's `/` route at the prefix itself (`/v1/local`), not at
    // `/v1/local/` — the trailing-slash form 404s, so the external index address
    // is the bare prefix.
    let ok = mounted()
        .oneshot(
            Request::builder()
                .uri("/v1/local")
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ok.status(), StatusCode::OK);

    // The gateway's admin headers alone are still refused through the nest.
    let refused = mounted()
        .oneshot(
            Request::builder()
                .uri("/v1/local")
                .header("x-tokera-user", "boss")
                .header("x-tokera-provider", "google")
                .header("x-tokera-email", "johnathan.sharratt@gmail.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(refused.status(), StatusCode::UNAUTHORIZED);
}

// =========================================================================
// The lift — the first situated namespace over the device (§C.1)
// =========================================================================

/// The shaft's landings, top-first-index. Used to place a body on a landing the
/// car is not resting at, so a call actually sets it in motion.
fn shaft(rt: &Arc<Runtime>) -> Vec<Where> {
    rt.hosted
        .get(WORLD)
        .expect("hosted")
        .read(|w| w.shaft().to_vec())
}

/// **`OPTIONS` returns the `use` body's `floor` enum, filled from live world
/// state.** The design's load-bearing claim (§11): the enum the schema hands
/// back is computed from `world.floor_names()` at OPTIONS time, so it is exactly
/// the set of levels the shaft serves — not a compiled-in list that could drift.
#[tokio::test]
async fn the_lift_schema_floor_enum_is_the_worlds_own_floor_names() {
    let (rt, tokens) = daemon();
    let landings = shaft(&rt);
    assert!(landings.len() >= 2, "the vault has a shaft");
    place_where(&rt, 5, landings[0].clone());

    let (status, body) = request(&rt, &tokens, 5, "OPTIONS", "/lift/command-shaft", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let expected = rt.hosted.get(WORLD).unwrap().read(|w| w.floor_names());
    let enum_floors: Vec<String> = body["methods"]["POST"]["body"]["properties"]["floor"]["enum"]
        .as_array()
        .expect("a floor enum")
        .iter()
        .map(|v| v.as_str().expect("a level name").to_string())
        .collect();
    assert_eq!(enum_floors, expected, "the enum is not the shaft's floors");

    // The rest of the schema is what §5.1 describes: the resource id and the
    // read shape the character can act against in one turn.
    assert_eq!(body["id"], json!("lift/command-shaft"));
    assert_eq!(body["methods"]["GET"]["returns"]["here"], json!("boolean"));
    assert_eq!(
        body["methods"]["POST"]["summary"],
        json!("Call or ride the lift.")
    );
}

/// **A body standing at a station is offered the station's verbs, each with the
/// body its act takes (§11).** Read off the real vault: the address the grammar
/// will let the character name is the address the route serves, and the body the
/// grammar pairs with it is the act's own parameters — `{}` at the command table,
/// whose `collect_mission` names nothing.
#[tokio::test]
async fn a_body_at_the_command_table_is_offered_its_mission_verb_with_an_empty_body() {
    use npcd::engine::tools::{specs_within, Mode};

    let (rt, _tokens) = daemon();
    place_where(&rt, 15, Where::new("vault-command", "command-room"));
    let table = instance_id("vault-command", "command-room", "order-table");
    let url = format!("http://local/command/{table}/collect_mission");

    let within = rt.within(15);
    assert!(
        within.invokable.iter().any(|i| i.url == url),
        "the table's verb is not offered: {:?}",
        within.invokable
    );

    let specs = specs_within(Mode::Physical, &within);
    let invoke = specs
        .iter()
        .find(|s| s.name == "invoke")
        .expect("there is something to invoke here");
    let address = invoke
        .params
        .iter()
        .find(|p| p.name == "url")
        .expect("a url parameter");
    assert!(
        address
            .enum_values
            .as_ref()
            .is_some_and(|u| u.contains(&url)),
        "the url enum omits {url}"
    );
    let (_, shape) = address
        .shapes
        .iter()
        .find(|(u, _)| *u == url)
        .expect("the address brings its body");
    assert_eq!(shape.len(), 1);
    assert_eq!(shape[0].name, "body");
    assert_eq!(
        shape[0].properties.as_ref().map(Vec::len),
        Some(0),
        "collect_mission takes no fields"
    );
}

/// **Every address a body is offered is one the router answers.** A body is stood
/// at every place in the vault and the resource behind each `invoke` address its
/// grammar would let it name is read over the socket: none may `404`, so a
/// character can never be handed an address the router has no route for — the
/// seat, blast door and wall turret all stand in the vault and afford nothing,
/// so offering them was a `404` each time.
#[tokio::test]
async fn every_address_a_body_is_offered_is_served() {
    use npc_map::load::MapSet;

    let (rt, tokens) = daemon();
    let set = MapSet::load_dir(ROOMS).expect("the vault loads");
    let mut offered = 0;
    let mut npc_id = 1000;
    for area in set.areas() {
        for node in &area.nodes {
            let at = Where::new(area.id.as_str(), node.id.as_str());
            if set.instances_at(&at).is_empty() {
                continue;
            }
            npc_id += 1;
            place_where(&rt, npc_id, at.clone());
            let within = rt.within(npc_id);
            for invokable in &within.invokable {
                let (resource, _) = invokable.url.rsplit_once('/').expect("a verb path");
                let path = resource
                    .strip_prefix("http://local")
                    .expect("a local address");
                let (status, body) = request(&rt, &tokens, npc_id, "GET", path, None).await;
                assert_eq!(
                    status,
                    StatusCode::OK,
                    "{} is offered at {at:?} but {resource} does not answer: {body}",
                    invokable.url
                );
                offered += 1;
            }
        }
    }
    assert!(offered > 0, "the sweep read nothing");
}

/// **A call from a landing lands.** Standing at the shaft with the car away,
/// `POST .../call` runs the real `lift_call` and comes back `200 {ok:true,…}`.
#[tokio::test]
async fn calling_the_lift_from_a_landing_is_accepted() {
    let (rt, _tokens) = daemon();
    let landings = shaft(&rt);
    let top = landings.len() - 1;
    place_where(&rt, 6, landings[top].clone());

    let (status, ack) = rt
        .effector_invoke(6, "/lift/command-shaft/call", json!({}), None)
        .await
        .expect("the fast path answers");
    assert_eq!(status, StatusCode::OK, "{ack}");
    assert_eq!(ack["ok"], json!(true), "{ack}");
    assert!(
        ack["detail"]
            .as_str()
            .is_some_and(|d| d.contains("call the lift")),
        "the world's own line did not come back: {ack}"
    );
}

/// **The consequence loop, end to end (§10).** From a landing the car is not at,
/// `invoke` the call — acknowledged now — then let the world tick; the car
/// travels on the clock and arrives, and the arrival is *observable through the
/// device*: the status that read `here:false` now reads `here:true`, boardable.
#[tokio::test]
async fn a_called_lift_arrives_over_ticks_and_the_status_shows_it_boardable() {
    let (rt, tokens) = daemon();
    let world = rt.hosted.get(WORLD).expect("hosted");
    let landings = shaft(&rt);
    let top = landings.len() - 1;
    place_where(&rt, 7, landings[top].clone());

    // The car is elsewhere: the status says it is not boardable here yet.
    let (read, before) = request(&rt, &tokens, 7, "GET", "/lift/command-shaft", None).await;
    assert_eq!(read, StatusCode::OK, "{before}");
    assert_eq!(
        before["here"],
        json!(false),
        "the car was already here, so the call proves nothing: {before}"
    );

    // Call it — acknowledged this turn; the travel is the world's to do.
    let (status, ack) = rt
        .effector_invoke(7, "/lift/command-shaft/call", json!({}), None)
        .await
        .expect("the fast path answers");
    assert_eq!(status, StatusCode::OK, "{ack}");

    // The world ticks; the car covers the shaft one floor per moment.
    let mut arrived = false;
    for _ in 0..80 {
        if world.read(|w| w.lift().unwrap().boardable_at(top)) {
            arrived = true;
            break;
        }
        world.with(|w| {
            w.tick();
        });
    }
    assert!(arrived, "the car never arrived over the tick loop");

    // And the character learns it through the device, not the return value: the
    // near-you status now reports the car boardable at this landing.
    let (read, after) = request(&rt, &tokens, 7, "GET", "/lift/command-shaft", None).await;
    assert_eq!(read, StatusCode::OK, "{after}");
    assert_eq!(
        after["here"],
        json!(true),
        "the arrival was not observable through the device: {after}"
    );
    assert_eq!(
        after["moving"],
        json!(false),
        "the car is still moving: {after}"
    );
}

/// **A wrong standpoint is refused, mapped to `409`.** Proximity is the enact
/// handler's to enforce (§C.1): a call from a work node, not a landing, comes
/// back as the refusal `lift_call` writes, in the estate error shape.
#[tokio::test]
async fn calling_the_lift_from_the_wrong_standpoint_is_refused() {
    let (rt, _tokens) = daemon();
    // Band one places terminals, not a landing.
    place(&rt, 8, "band-one");

    let (status, refused) = rt
        .effector_invoke(8, "/lift/command-shaft/call", json!({}), None)
        .await
        .expect("the fast path answers");
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"], json!("refused"), "{refused}");
    assert!(
        refused["detail"]
            .as_str()
            .is_some_and(|d| d.contains("not at the lift")),
        "the refusal's own words were not carried through: {refused}"
    );
}

// =========================================================================
// The generic station routes — every part namespace's verbs at once (§9, §12)
// =========================================================================

/// The instance id of the first `part` standing at `area/node` in the shipped
/// vault, as `<part>~<ordinal>` — derived from the map rather than hard-coded, so
/// the tests never spell an ordinal the map is free to decide (see
/// [`npc_map::instance`]).
fn instance_id(area: &str, node: &str, part: &str) -> String {
    use npc_map::load::MapSet;
    MapSet::load_dir(ROOMS)
        .expect("the vault loads")
        .instances_at(&Where::new(area, node))
        .into_iter()
        .find(|i| i.part_id() == part)
        .unwrap_or_else(|| panic!("no `{part}` stands at `{area}/{node}`"))
        .id()
}

/// The instance ids the station tests address, each a real placement in the
/// shipped vault, resolved once from the map.
static CHRONICLE_TERMINAL: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    instance_id("vault-chronicle", "early-range", "chronicle-terminal")
});
static ACCESSION_DESK: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| instance_id("vault-command", "receiving", "accession-desk"));
static APPRAISAL_BENCH: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| instance_id("vault-chronicle", "sorting-room", "appraisal-bench"));
static EASEL: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| instance_id("vault-portraits", "north-studio", "easel"));

/// The verb keys an `OPTIONS` answer advertises under `POST`, sorted.
fn verbs(body: &Value) -> Vec<String> {
    let mut names: Vec<String> = body["methods"]["POST"]["verbs"]
        .as_object()
        .expect("a verbs object")
        .keys()
        .cloned()
        .collect();
    names.sort();
    names
}

/// **`OPTIONS` on a chronicle terminal names its verbs and their body shapes.**
/// The schema is built from the `Tool` catalogue — the acts whose `at` names
/// `chronicle-terminal` — so it lists exactly `add_entry`/`rewrite_page`/
/// `retire_entry`, and `add_entry`'s body is the raw two-string object its params
/// declare (`station.rs` `CHRONICLE_ADD_ENTRY`).
#[tokio::test]
async fn the_chronicle_options_lists_its_verbs_and_their_shapes() {
    let (rt, tokens) = daemon();
    place_where(&rt, 10, Where::new("vault-chronicle", "early-range"));

    let (status, body) = request(
        &rt,
        &tokens,
        10,
        "OPTIONS",
        &format!("/chronicle/{}", *CHRONICLE_TERMINAL),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    assert_eq!(
        body["id"],
        json!(format!("chronicle/{}", *CHRONICLE_TERMINAL))
    );
    assert!(
        body["summary"]
            .as_str()
            .is_some_and(|s| s.contains("A terminal onto the world's recorded history")),
        "the part's long description is the schema summary: {body}"
    );
    // Every act the catalogue attaches to a chronicle terminal: the three
    // chronicle verbs, plus the bench working-loop and file verbs that also name
    // this station in their `at` (§9, Appendix C.25/C.26). The set is raw — the
    // catalogue's whole answer for this part.
    assert_eq!(
        verbs(&body),
        [
            "add_entry",
            "bench_blame",
            "bench_branch",
            "bench_commit",
            "bench_diff",
            "bench_log",
            "bench_restore",
            "bench_stage",
            "bench_stash",
            "bench_stash_pop",
            "bench_status",
            "bench_unstage",
            "file_edit",
            "file_list",
            "file_read",
            "file_write",
            "retire_entry",
            "rewrite_page",
        ],
        "the schema is the catalogue's acts for this part: {body}"
    );
    // The body schema is raw, built from the act's own params — two required
    // strings — never invented.
    assert_eq!(
        body["methods"]["POST"]["verbs"]["add_entry"]["body"],
        json!({
            "type": "object",
            "properties": { "to": { "type": "string" }, "what": { "type": "string" } },
            "required": ["to", "what"]
        }),
        "{body}"
    );
}

/// **`GET` reads the station as a character does.** The descriptor names the part
/// it is, what it is called, and the short verbs it affords here — the same set
/// `OPTIONS` describes.
#[tokio::test]
async fn a_get_reads_the_station_descriptor() {
    let (rt, tokens) = daemon();
    place_where(&rt, 11, Where::new("vault-chronicle", "early-range"));

    let (status, body) = request(
        &rt,
        &tokens,
        11,
        "GET",
        &format!("/chronicle/{}", *CHRONICLE_TERMINAL),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["id"],
        json!(format!("chronicle/{}", *CHRONICLE_TERMINAL))
    );
    assert_eq!(body["part"], json!("chronicle-terminal"));
    assert_eq!(body["name"], json!("world history terminal"));
    // In catalogue order: the chronicle verbs, then the bench and file verbs the
    // same station affords.
    assert_eq!(
        body["verbs"],
        json!([
            "add_entry",
            "rewrite_page",
            "retire_entry",
            "bench_branch",
            "bench_diff",
            "bench_stash",
            "bench_stash_pop",
            "bench_restore",
            "bench_stage",
            "bench_unstage",
            "bench_commit",
            "bench_status",
            "bench_blame",
            "bench_log",
            "file_read",
            "file_write",
            "file_edit",
            "file_list",
        ])
    );
}

/// **A chronicle verb runs the real act, and the world's own refusal comes
/// back.** `POST .../add_entry` synthesises `chronicle_add_entry` and runs it
/// through the dispatch; the seeded "third era" is `Filed`, and the shipped
/// handler (`work.rs` `write_into` → `record.write`) refuses a write over a
/// filed record — a `409` carrying that refusal's own words, exactly as the
/// world would answer the act reached any other way.
#[tokio::test]
async fn posting_a_chronicle_verb_runs_the_real_act_and_carries_its_refusal() {
    let (rt, tokens) = daemon();
    place_where(&rt, 12, Where::new("vault-chronicle", "early-range"));

    let (status, refused) = request(
        &rt,
        &tokens,
        12,
        "POST",
        &format!("/chronicle/{}/add_entry", *CHRONICLE_TERMINAL),
        Some(json!({ "to": "the third era", "what": "the second burning, and that nobody rebuilt after" })),
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

/// **The record namespace: `OPTIONS` at the accession desk, then `accession`
/// lands.** The desk affords the custody verbs, and `POST .../accession` runs
/// the real `record_accession` against the seeded intake — a `200 {ok:true}`
/// with the world's own line, because the shipped handler writes the origin and
/// succeeds.
#[tokio::test]
async fn the_record_accession_options_and_runs() {
    let (rt, tokens) = daemon();
    place_where(&rt, 13, Where::new("vault-command", "receiving"));

    let (status, schema) = request(
        &rt,
        &tokens,
        13,
        "OPTIONS",
        &format!("/record/{}", *ACCESSION_DESK),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{schema}");
    assert_eq!(verbs(&schema), ["accession", "hand_on", "write_provenance"]);
    assert_eq!(
        schema["methods"]["POST"]["verbs"]["accession"]["body"],
        json!({
            "type": "object",
            "properties": { "what": { "type": "string" }, "from": { "type": "string" } },
            "required": ["what", "from"]
        }),
        "{schema}"
    );

    let (status, ack) = request(
        &rt,
        &tokens,
        13,
        "POST",
        &format!("/record/{}/accession", *ACCESSION_DESK),
        Some(json!({ "what": "the western intake", "from": "a caravan that could not say who packed it" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{ack}");
    assert_eq!(ack["ok"], json!(true), "{ack}");
    assert!(
        ack["detail"]
            .as_str()
            .is_some_and(|d| d.contains("taken in")),
        "the world's own line did not come back: {ack}"
    );
}

/// **The same namespace, another part: `appraise` at the appraisal bench.** The
/// bench and the desk share the `record` namespace but afford different verbs;
/// `POST .../appraise` runs `record_appraise`, which the shipped handler always
/// records as a verdict — a `200 {ok:true}`.
#[tokio::test]
async fn posting_record_appraise_at_the_bench_runs() {
    let (rt, tokens) = daemon();
    place_where(&rt, 14, Where::new("vault-chronicle", "sorting-room"));

    let (status, schema) = request(
        &rt,
        &tokens,
        14,
        "OPTIONS",
        &format!("/record/{}", *APPRAISAL_BENCH),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{schema}");
    assert!(
        verbs(&schema).contains(&"appraise".to_string()),
        "the bench affords appraise: {schema}"
    );

    let (status, ack) = request(
        &rt,
        &tokens,
        14,
        "POST",
        &format!("/record/{}/appraise", *APPRAISAL_BENCH),
        Some(json!({ "what": "the western intake", "verdict": "nothing depends on it and nothing ever has" })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{ack}");
    assert_eq!(ack["ok"], json!(true), "{ack}");
    assert!(
        ack["detail"].as_str().is_some_and(|d| d.contains("weigh")),
        "the world's own line did not come back: {ack}"
    );
}

/// **The portrait namespace: `OPTIONS` at an easel, then `draw`.** The easel
/// affords `draw`/`prompt_read`/`prompt_edit`; `POST .../draw` runs
/// `portrait_draw`. This world was stood up with no mind folder behind its
/// benches, so the shipped handler refuses — the bench has no documents to draw
/// into — and that refusal comes back as a `409` in the world's own words,
/// pinning that drawing runs through the real bench dispatch.
#[tokio::test]
async fn the_portrait_options_and_draw_runs_through_the_bench() {
    let (rt, tokens) = daemon();
    place_where(&rt, 15, Where::new("vault-portraits", "north-studio"));

    let (status, schema) = request(
        &rt,
        &tokens,
        15,
        "OPTIONS",
        &format!("/portrait/{}", *EASEL),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{schema}");
    // The easel's portrait verbs, plus the bench/file working surface it shares
    // with the other authoring stations.
    assert_eq!(
        verbs(&schema),
        [
            "bench_blame",
            "bench_branch",
            "bench_commit",
            "bench_diff",
            "bench_log",
            "bench_restore",
            "bench_stage",
            "bench_stash",
            "bench_stash_pop",
            "bench_status",
            "bench_unstage",
            "draw",
            "file_edit",
            "file_list",
            "file_read",
            "file_write",
            "prompt_edit",
            "prompt_read",
        ]
    );

    let (status, refused) = request(
        &rt,
        &tokens,
        15,
        "POST",
        &format!("/portrait/{}/draw", *EASEL),
        Some(json!({ "of": "ash-the-drifter", "carrying": "somebody long believed about a thing they got wrong" })),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"], json!("refused"), "{refused}");
    assert!(
        refused["detail"]
            .as_str()
            .is_some_and(|d| d.contains("no documents")),
        "the bench's own refusal was not carried through: {refused}"
    );
}

/// **An unknown verb at a real station is a `404`.** `frobnicate` reconstructs to
/// no catalogue act that attaches to a chronicle terminal, so the resource does
/// not afford it — the estate error shape, `unknown_verb`.
#[tokio::test]
async fn an_unknown_verb_at_a_station_is_not_found() {
    let (rt, tokens) = daemon();
    place_where(&rt, 16, Where::new("vault-chronicle", "early-range"));

    let (status, body) = request(
        &rt,
        &tokens,
        16,
        "POST",
        &format!("/chronicle/{}/frobnicate", *CHRONICLE_TERMINAL),
        Some(json!({ "to": "the third era", "what": "anything" })),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"], json!("unknown_verb"), "{body}");
}

/// **A malformed or unresolvable id is a `404`.** An ordinal past the six
/// terminals that stand in the early range names nothing the map lays out, so it
/// resolves to nothing — refused before any act is synthesised.
#[tokio::test]
async fn a_malformed_instance_id_is_not_found() {
    let (rt, tokens) = daemon();
    place_where(&rt, 17, Where::new("vault-chronicle", "early-range"));

    let (status, body) = request(
        &rt,
        &tokens,
        17,
        "OPTIONS",
        "/chronicle/chronicle-terminal~99",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"], json!("not_found"), "{body}");
}

/// **A station out of reach is a `404` — reach is the gate (§7.1, §8.3).** The
/// body stands in casting, a real chronicle terminal stands two levels away; an
/// as-npc token reaches only what is within its standpoint, so the terminal that
/// really exists is not addressable from here.
#[tokio::test]
async fn a_station_out_of_reach_is_not_found() {
    let (rt, tokens) = daemon();
    // A real terminal, a body nowhere near it.
    place(&rt, 18, "band-one");

    let (status, body) = request(
        &rt,
        &tokens,
        18,
        "OPTIONS",
        &format!("/chronicle/{}", *CHRONICLE_TERMINAL),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"], json!("not_found"), "{body}");
}

// =========================================================================
// The personal routes — `/phone` and `/history` (§7.2, C.27)
// =========================================================================

/// **`GET /phone` is a real read of the body's threads, empty when it has
/// none.** A body freshly placed into the bare test daemon is on no
/// conversation — nothing joins it to the channel here — so its listing is an
/// empty array rather than an error: a handset that has reached nobody.
#[tokio::test]
async fn the_phone_lists_a_placed_bodys_threads() {
    let (rt, tokens) = daemon();
    place(&rt, 20, "band-one");

    let (status, body) = request(&rt, &tokens, 20, "GET", "/phone", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["threads"],
        json!([]),
        "a body on no conversation lists an empty phone: {body}"
    );
}

/// **`OPTIONS /phone` lists the phone verbs and their raw body shapes.** One
/// `POST` verb per phone act, each built from the act's own params — so it names
/// exactly `message`/`invite`/`open_group`/`reach_out`, plus `sign_off`
/// (addressed per-thread), and `message`'s body is the raw two-string object its
/// params declare (`acts.rs` `MESSAGE`). `send_image` is **not** advertised: it
/// is performed above the body layer and is not routable through `body::perform`,
/// so mounting it would be a route that only ever `500`s.
#[tokio::test]
async fn the_phone_options_lists_its_verbs_and_their_shapes() {
    let (rt, tokens) = daemon();
    place(&rt, 21, "band-one");

    let (status, body) = request(&rt, &tokens, 21, "OPTIONS", "/phone", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], json!("phone"));

    // The phone acts, raw — the catalogue's whole answer for the phone surface,
    // minus `send_image`, which is deferred (performed above the body layer).
    assert_eq!(
        verbs(&body),
        ["invite", "message", "open_group", "reach_out", "sign_off"],
        "the schema is the phone acts' verbs: {body}"
    );
    assert!(
        !verbs(&body).contains(&"send_image".to_string()),
        "send_image must not be advertised — it is not routable through body::perform: {body}"
    );
    // `message`'s body is raw, from its own params — two required strings.
    assert_eq!(
        body["methods"]["POST"]["verbs"]["message"]["body"],
        json!({
            "type": "object",
            "properties": { "to": { "type": "string" }, "intent": { "type": "string" } },
            "required": ["to", "intent"]
        }),
        "{body}"
    );
    // `sign_off` is addressed per-thread: its `to` (the thread) comes from the
    // path, so its body drops `to` and keeps only the parting `intent`.
    assert_eq!(
        body["methods"]["POST"]["verbs"]["sign_off"]["addressed"],
        json!("per-thread"),
        "{body}"
    );
    assert_eq!(
        body["methods"]["POST"]["verbs"]["sign_off"]["body"],
        json!({
            "type": "object",
            "properties": { "intent": { "type": "string" } },
            "required": ["intent"]
        }),
        "{body}"
    );
}

/// **`POST /phone/message` runs the real act, and the world's own refusal comes
/// back.** In the bare test daemon the body is on no thread, so `message`
/// synthesised and run through the dispatch refuses — `s.threads.send` finds no
/// conversation by that name — and that refusal is a `409` carrying the
/// handler's own words, exactly as the act would answer reached any other way.
#[tokio::test]
async fn posting_a_phone_message_runs_the_real_act_and_carries_its_refusal() {
    let (rt, tokens) = daemon();
    place(&rt, 22, "band-one");

    let (status, refused) = request(
        &rt,
        &tokens,
        22,
        "POST",
        "/phone/message",
        Some(json!({ "to": "the channel", "intent": "where each of you is working" })),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"], json!("refused"), "{refused}");
    assert!(
        refused["detail"]
            .as_str()
            .is_some_and(|d| d.contains("no conversation")),
        "the handler's own refusal was not carried through: {refused}"
    );
}

/// **An unknown phone verb is a `404`.** `frobnicate` is not one of the phone
/// acts, so it is not something the phone affords — the estate error shape,
/// `unknown_verb`.
#[tokio::test]
async fn an_unknown_phone_verb_is_not_found() {
    let (rt, tokens) = daemon();
    place(&rt, 23, "band-one");

    let (status, body) = request(
        &rt,
        &tokens,
        23,
        "POST",
        "/phone/frobnicate",
        Some(json!({ "to": "anyone", "intent": "anything" })),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"], json!("unknown_verb"), "{body}");
}

/// **`GET /history` reads the body's witnessed past, and reads it without
/// spending it.** A second body arrives beside the first and says something in
/// the same room; the first body's history reports it (narrowed to what it could
/// make out, `witness::since`). And the read is non-destructive: called twice it
/// returns the same events, because it peeks the log rather than marking it
/// seen — the delta the engine relies on is untouched.
#[tokio::test]
async fn history_reads_what_was_witnessed_without_disturbing_engine_state() {
    let (rt, tokens) = daemon();
    // Two bodies in one room. The first is the reader; the second acts, so there
    // is something for the first to have witnessed.
    place(&rt, 24, "band-one");
    place(&rt, 25, "band-one");

    // The second body says something, in the same room — a witnessable event.
    let world = rt.hosted.get(WORLD).expect("hosted");
    let speaker = Runtime::body_id(25);
    world.with(|w| {
        w.say(&speaker, "the redoubt burned twice").expect("said");
    });

    let (status, body) = request(&rt, &tokens, 24, "GET", "/history", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = body["events"].as_array().expect("an events array");
    assert!(
        events
            .iter()
            .any(|e| e["what"].as_str().is_some_and(|w| w.contains("redoubt"))),
        "the witnessed line did not appear in the history: {body}"
    );

    // Reading it again returns the same events — the peek advanced no cursor, so
    // the engine's view of what this body has yet to see is unchanged.
    let (status_again, body_again) = request(&rt, &tokens, 24, "GET", "/history", None).await;
    assert_eq!(status_again, StatusCode::OK, "{body_again}");
    assert_eq!(
        body_again["events"], body["events"],
        "history was consumed on read: the second call differs from the first"
    );
}
