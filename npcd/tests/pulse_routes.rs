//! The per-character write routes, called the way a console calls them.
//!
//! `/v1/npc/:nid/direct`, `/v1/npc/:nid/pulse`, `/v1/npc/:nid/window` and the
//! mission routes under `/v1/npc/:nid/mission` all name a character in the
//! path. The wire carries a character id as a base-36 string
//! ([`npc_id_wire`]), so the one thing every test here does differently from a
//! unit test of a handler is **address the character by that string** — a route
//! that parses the segment as a decimal `u64` answers 400 for every real id, and
//! no test that built the id as a number could see it.
//!
//! The router is the daemon's own ([`npcd::engine::api`]) over a real
//! [`Authored`], a real [`Runtime`] and a real scheduler; only the decode is
//! absent, because nothing here starts a character task. Delivery is observed
//! by ticking the scheduler by hand and reading what the character was handed.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use npcd::accounts::Accounts;
use npcd::api::Authored;
use npcd::collections::Libraries;
use npcd::engine::runtime::Runtime;
use npcd::images::Images;
use npcd::mind::Mind;
use npcd::npcs::{npc_id_of_wire, npc_id_wire, Npcs};
use npcd::projection::Source;
use npcd::registry::Registry;
use serde_json::{json, Value};
use tower::ServiceExt;
use web::auth::session::Identity;

/// The subject treated as an admin; every other subject is an ordinary user.
const ADMIN: &str = "google-admin";

fn tmp(tag: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let p = std::env::temp_dir().join(format!(
        "npcd-pulse-routes-{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn identity(sub: &str) -> Identity {
    Identity {
        provider: "google".into(),
        sub: sub.into(),
        email: "wren@example.com".into(),
        name: "Wren S".into(),
        picture: String::new(),
        exp: 0,
    }
}

/// A daemon with one character owned by `google-owner`.
struct Rig {
    app: Router,
    rt: Arc<Runtime>,
    /// The character, as the wire names it.
    wire: String,
    nid: u64,
}

impl Rig {
    /// `awake` puts the character in the scheduler; without it the character
    /// exists in the registry and nowhere else, which is the daemon's state
    /// between a character being created and the engine finishing its load.
    async fn new(awake: bool) -> Self {
        let base = tmp("rig");
        let owner_identity = identity("google-owner");
        let mut accounts = Accounts::load(base.join("accounts")).unwrap();
        let owner = accounts.upsert(&owner_identity, 0).unwrap()["user_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let mut npcs = Npcs::load(&base).unwrap();
        let made = npcs
            .create(
                &owner_identity,
                &owner,
                &json!({ "name": "Varek", "world_id": "battle-cities", "personality_id": "commander" }),
                0,
            )
            .unwrap();
        let wire = made["npc_id"].as_str().unwrap().to_owned();
        let nid = npc_id_of_wire(&wire).unwrap();
        assert_eq!(npc_id_wire(nid), wire);

        let rt = Runtime::new(Mind::new(None), &base);
        if awake {
            rt.scheduler.wake(nid, 0, 0);
        }
        let state = Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            accounts,
            npcs,
            serde_yaml::from_str(&format!("admins:\n  - sub: {ADMIN}\n")).unwrap(),
            Libraries::load(&Source::resolve(None).unwrap()),
            Mind::new(None),
            Images::new(&base),
            Default::default(),
        )
        .with_runtime(rt.clone());
        let app = npcd::engine::api(state.clone()).into_router(state);
        Self { app, rt, wire, nid }
    }

    async fn call(
        &self,
        method: &str,
        path: &str,
        sub: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let id = identity(sub);
        let mut b = Request::builder()
            .method(method)
            .uri(path)
            .header("x-tokera-user", &id.sub)
            .header("x-tokera-provider", &id.provider)
            .header("x-tokera-email", &id.email)
            .header("x-tokera-name", &id.name);
        let req = match body {
            Some(v) => {
                b = b.header("content-type", "application/json");
                b.body(Body::from(v.to_string())).unwrap()
            }
            None => b.body(Body::empty()).unwrap(),
        };
        let res = self.app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    async fn post(&self, path: &str, sub: &str, body: Value) -> (StatusCode, Value) {
        self.call("POST", path, sub, Some(body)).await
    }

    async fn get(&self, path: &str, sub: &str) -> (StatusCode, Value) {
        self.call("GET", path, sub, None).await
    }

    fn path(&self, tail: &str) -> String {
        format!("/v1/npc/{}/{tail}", self.wire)
    }

    /// What the character is handed on its next tick, as prose.
    fn heard(&self) -> Vec<String> {
        let mut prose = Vec::new();
        self.rt.scheduler.tick(self.nid, 1_000, 0, |events, _| {
            prose = events.iter().map(|e| e.prose()).collect();
            Vec::new()
        });
        prose
    }
}

const OWNER: &str = "google-owner";
const STRANGER: &str = "google-stranger";

#[tokio::test]
async fn the_wire_id_of_a_character_is_not_a_decimal_number() {
    let rig = Rig::new(true).await;
    assert!(
        rig.wire.parse::<u64>().is_err(),
        "a test id that happens to be all digits cannot tell a base-36 path from a decimal one: {}",
        rig.wire
    );
}

#[tokio::test]
async fn a_direct_line_reaches_the_character_it_is_addressed_to() {
    let rig = Rig::new(true).await;
    let (status, body) = rig
        .post(
            &rig.path("direct"),
            OWNER,
            json!({ "text": "  hold the east gate  " }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["delivered"], true);
    assert_eq!(body["preempts"], true);
    let prose = body["prose"].as_str().unwrap();
    assert!(prose.contains("Wren S"), "spoken by the operator: {prose}");
    assert!(prose.contains("hold the east gate"), "{prose}");

    let heard = rig.heard();
    assert_eq!(heard.len(), 1, "{heard:?}");
    assert_eq!(heard[0], prose);
}

#[tokio::test]
async fn a_direct_line_can_name_its_speaker_and_how_loudly_it_lands() {
    let rig = Rig::new(true).await;
    let (status, body) = rig
        .post(
            &rig.path("direct"),
            OWNER,
            json!({ "text": "report in", "speaker": "Command", "salience": 0.2 }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["preempts"], false);
    assert!((body["salience"].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert!(body["prose"].as_str().unwrap().contains("Command"));
}

#[tokio::test]
async fn a_blank_direct_line_is_refused_and_delivers_nothing() {
    let rig = Rig::new(true).await;
    let (status, body) = rig
        .post(&rig.path("direct"), OWNER, json!({ "text": "   " }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "empty_message");
    assert!(rig.heard().is_empty());
}

#[tokio::test]
async fn an_injected_command_is_parsed_and_delivered() {
    let rig = Rig::new(true).await;
    let (status, body) = rig
        .post(
            &rig.path("pulse"),
            OWNER,
            json!({ "line": "/say the gate is open" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["delivered"], true);
    assert_eq!(body["command"], "say");
    assert!(body["prose"].as_str().unwrap().contains("the gate is open"));
    assert_eq!(rig.heard().len(), 1);
}

#[tokio::test]
async fn a_mistyped_command_is_a_400_and_never_speech() {
    let rig = Rig::new(true).await;
    let (status, body) = rig
        .post(&rig.path("pulse"), OWNER, json!({ "line": "/hrut" }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "bad_command");
    assert!(rig.heard().is_empty());
}

#[tokio::test]
async fn the_window_starts_empty_and_holds_what_the_character_was_told() {
    let rig = Rig::new(true).await;
    let (status, body) = rig.get(&rig.path("window"), OWNER).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["empty"], true);
    assert_eq!(body["turns"], json!([]));
    assert!(body["cap"].as_u64().unwrap() > 0);

    rig.post(
        &rig.path("direct"),
        OWNER,
        json!({ "text": "hold the east gate" }),
    )
    .await;
    rig.heard();

    let (status, body) = rig.get(&rig.path("window"), OWNER).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["empty"], false);
    let turns = body["turns"].as_array().unwrap();
    assert!(
        turns
            .iter()
            .any(|t| t["text"].as_str().unwrap().contains("hold the east gate")),
        "{turns:?}"
    );
}

#[tokio::test]
async fn a_character_that_is_not_in_the_scheduler_answers_503() {
    let rig = Rig::new(false).await;
    let (status, body) = rig
        .post(&rig.path("direct"), OWNER, json!({ "text": "hello" }))
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "not_awake");
    let (status, body) = rig
        .post(&rig.path("pulse"), OWNER, json!({ "line": "/say hello" }))
        .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "not_awake");
    let (status, body) = rig.get(&rig.path("window"), OWNER).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "not_awake");
}

#[tokio::test]
async fn somebody_elses_character_is_a_404_on_every_write_and_read() {
    let rig = Rig::new(true).await;
    for (method, tail, body) in [
        ("POST", "direct", Some(json!({ "text": "hi" }))),
        ("POST", "pulse", Some(json!({ "line": "/say hi" }))),
        ("GET", "window", None),
        ("POST", "mission", Some(json!({ "prompt": "go" }))),
        ("GET", "mission", None),
        ("POST", "mission/cancel", Some(json!({}))),
    ] {
        let (status, _) = rig.call(method, &rig.path(tail), STRANGER, body).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{method} {tail}");
    }
    assert!(
        rig.heard().is_empty(),
        "nothing reached a character its caller does not own"
    );
}

#[tokio::test]
async fn an_id_that_names_no_character_is_a_404() {
    let rig = Rig::new(true).await;
    let (status, _) = rig
        .post(
            "/v1/npc/zzzzzzzzzzzzz/direct",
            OWNER,
            json!({ "text": "hi" }),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = rig.get("/v1/npc/not-an-id!/window", OWNER).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_mission_is_lodged_read_and_cancelled_by_wire_id() {
    let rig = Rig::new(true).await;

    let (status, body) = rig.get(&rig.path("mission"), OWNER).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["on_mission"], false);

    let (status, body) = rig
        .post(&rig.path("mission"), OWNER, json!({ "prompt": "   " }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "empty_mission");

    // Past the id and the brief, the next refusal is the honest one: this
    // character has no body in a world to carry it.
    let (status, body) = rig
        .post(&rig.path("mission"), OWNER, json!({ "prompt": "scout" }))
        .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"], "not_in_a_world");

    let (status, body) = rig
        .post(&rig.path("mission/cancel"), OWNER, json!({}))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["cancelled"], false);
}
