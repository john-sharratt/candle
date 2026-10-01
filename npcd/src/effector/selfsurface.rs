//! `http://local/self` — the character's own maintained state, as a read view
//! (effector design §7.2, Appendix C.27).
//!
//! A character reaches its own plan, orders, beliefs and memory wherever it
//! stands — these are the character's own, not the room's, so they carry no
//! instance id and no reach check (§7.2). Every read here is **read-only**: the
//! writes happen at the situated stations that own them (C.11, C.21) and project
//! forward. Those writes land in the agency plane through
//! [`crate::effector::plan`]; this is the window onto what they wrote.
//!
//! # It reads through the shared cast, and never the world lock
//!
//! The one handle these routes touch is the runtime's shared
//! [`Npcs`](crate::npcs::Npcs) — the same cast the console API and the load walk
//! use ([`crate::engine::runtime::Runtime::npcs`]). The character's own record is
//! resolved by the caller's npc id ([`crate::npcs::Npcs::payload`], owner-blind:
//! the effector is the character's own body), the projection is rendered while
//! the cast's read lock is held, and the lock is released before the response is
//! built. The world lock is never taken, so a `/self` read can never contend with
//! a tick.
//!
//! # Degrades, never panics
//!
//! A runtime that has been dropped, a cast handle that was never installed (a
//! test path with no `set_npcs`), or a character with nothing written all answer
//! an **empty** projection rather than an error — the read is always safe to make
//! (§7.2: always available).
//!
//! # Plan and orders are one layer; memory is not on the record
//!
//! `plan` and `orders` both render the substrate **agency** layer
//! ([`AuthoredStrategy`](candle_conversation::persistence::record::AuthoredStrategy)):
//! the record stores one `agency` vector, and `plan_*`/`orders_*` both write it
//! (§9.2) — there is no separate ledger-orders layer on `NpcPayload`, so orders
//! is the same layer the plan reads. `memory` has **no layer on the record** at
//! all — a character's memory lives in the substrate's own records and the mind's
//! `layers/memory/`, not on `NpcPayload` — so `/self/memory` answers an empty
//! projection here; the body's *witnessed* recent past is the separate
//! [`/history`](crate::effector::history) surface.

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Extension, Json, Router};
use serde_json::{json, Value};

use candle_conversation::persistence::record::NpcPayload;

use crate::effector::auth::DeviceCaller;
use crate::effector::router::Local;
use crate::npcs;

/// The `/self` sub-router: an index and the four read views, all `GET`.
///
/// Returned without state so the outer router supplies its [`Local`] and the
/// device-auth layer, exactly as the station routes are mounted.
pub fn routes() -> Router<Local> {
    Router::new()
        .route("/", get(index))
        .route("/plan", get(plan))
        .route("/orders", get(orders))
        .route("/beliefs", get(beliefs))
        .route("/memory", get(memory))
}

/// `GET /self` — what the character can read about itself, one url each.
async fn index() -> Response {
    Json(json!({
        "routes": [
            route("http://local/self/plan", "the plan you are working to"),
            route("http://local/self/orders", "the orders you hold"),
            route("http://local/self/beliefs", "what you believe"),
            route("http://local/self/memory", "what you remember"),
        ]
    }))
    .into_response()
}

fn route(url: &str, summary: &str) -> Value {
    json!({ "url": url, "summary": summary })
}

/// `GET /self/plan` — the character's own strategies, from the agency layer.
async fn plan(State(local): State<Local>, Extension(caller): Extension<DeviceCaller>) -> Response {
    read_self(&local, &caller, npcs::agency_wire, json!({ "agency": [] })).await
}

/// `GET /self/orders` — the same agency layer the plan reads (§9.2): the record
/// holds one `agency` vector, and `orders_*` write it too.
async fn orders(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
) -> Response {
    read_self(&local, &caller, npcs::agency_wire, json!({ "agency": [] })).await
}

/// `GET /self/beliefs` — what the operator (and the sleep clock) has stated the
/// character believes.
async fn beliefs(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
) -> Response {
    read_self(
        &local,
        &caller,
        npcs::beliefs_wire,
        json!({ "beliefs": [] }),
    )
    .await
}

/// `GET /self/memory` — empty here: `NpcPayload` carries no memory layer (see the
/// module docs). The body's witnessed recent past is `/history`.
async fn memory(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
) -> Response {
    read_self(
        &local,
        &caller,
        |_| json!({ "memory": [] }),
        json!({ "memory": [] }),
    )
    .await
}

/// Render the caller's own record through `view`, or answer `empty`.
///
/// The cast's read lock is the only lock taken, and it is released the moment the
/// projection is built (the view owns its output). No world lock, no `.await`
/// while a lock is held past this point.
async fn read_self(
    local: &Local,
    caller: &DeviceCaller,
    view: fn(&NpcPayload) -> Value,
    empty: Value,
) -> Response {
    let Some(runtime) = local.runtime() else {
        return Json(empty).into_response();
    };
    let Some(npcs) = runtime.npcs() else {
        return Json(empty).into_response();
    };
    let rendered = {
        let cast = npcs.read().await;
        cast.payload(caller.npc_id).map(view)
    };
    Json(rendered.unwrap_or(empty)).into_response()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::header::AUTHORIZATION;
    use axum::http::{Request, StatusCode};
    use candle_conversation::persistence::record::{
        AuthoredBelief, AuthoredStrategy, Modulation, NpcPayload,
    };
    use tokio::sync::RwLock as AsyncRwLock;
    use tower::ServiceExt;

    use crate::effector::router::{router, Local};
    use crate::effector::token::{Scope, Tokens};
    use crate::engine::runtime::Runtime;
    use crate::mind::Mind;
    use crate::npcs::Npcs;

    const OWNER: &str = "u_1a2b3c4d";

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-self-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// A record spelled out — `NpcPayload` has no `Default` on purpose (every
    /// field is a decision), so a seed states each one, mirroring `persona.rs`.
    fn payload(
        npc_id: u64,
        agency: Vec<AuthoredStrategy>,
        beliefs: Vec<AuthoredBelief>,
    ) -> NpcPayload {
        NpcPayload {
            npc_id,
            owner_id: OWNER.into(),
            revision: 1,
            created_ms: 0,
            updated_ms: 0,
            state: "idle".into(),
            name: "Varek".into(),
            world_id: "battle-cities".into(),
            personality_id: "commander".into(),
            hidden: false,
            heartbeat_ms: 120_000,
            salience_gate: 0.5,
            tags: Vec::new(),
            persona_description: "A quartermaster.".into(),
            persona_origin: "authored".into(),
            portrait_image_id: None,
            portrait_origin: None,
            at: None,
            mood: None,
            beliefs,
            relationships: Vec::new(),
            agency,
            modulation: Modulation::default(),
        }
    }

    fn strat(id: &str, statement: &str, state: &str) -> AuthoredStrategy {
        AuthoredStrategy {
            strategy_id: id.into(),
            statement: statement.into(),
            parent_id: None,
            state: state.into(),
        }
    }

    fn belief(id: &str, statement: &str) -> AuthoredBelief {
        AuthoredBelief {
            belief_id: id.into(),
            statement: statement.into(),
            confidence: 0.8,
            threshold: 0.5,
        }
    }

    /// A runtime with a token store and, optionally, a seeded cast installed. No
    /// world hosted: `/self` reads touch only the shared cast.
    fn daemon(seed: Option<Vec<NpcPayload>>) -> (Arc<Runtime>, Arc<Tokens>) {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
        let tokens = Arc::new(Tokens::load(tmp()).expect("a fresh token store"));
        rt.set_tokens(tokens.clone());
        if let Some(payloads) = seed {
            let mut npcs = Npcs::load(&tmp()).expect("a fresh cast");
            npcs.import(payloads).expect("seeded");
            rt.set_npcs(Arc::new(AsyncRwLock::new(npcs)));
        }
        (rt, tokens)
    }

    async fn get(
        rt: &Arc<Runtime>,
        tokens: &Arc<Tokens>,
        npc_id: u64,
        path: &str,
    ) -> serde_json::Value {
        let token = tokens.mint(npc_id, Scope::AsNpc).expect("minted");
        let app = router(Local::new(tokens.clone(), rt));
        let req = Request::builder()
            .uri(path)
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK, "{path} did not answer 200");
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// **`/self/plan` returns the character's own strategies**, from the agency
    /// layer of its record, read through the shared cast.
    #[tokio::test]
    async fn self_plan_returns_the_agency_strategies() {
        let npc = 42;
        let (rt, tokens) = daemon(Some(vec![payload(
            npc,
            vec![strat("s0", "Get the ledger out of the district.", "active")],
            vec![],
        )]));
        let body = get(&rt, &tokens, npc, "/self/plan").await;
        let agency = body["agency"].as_array().expect("an agency array");
        assert_eq!(agency.len(), 1);
        assert_eq!(agency[0]["strategy_id"], "s0");
        assert_eq!(
            agency[0]["statement"],
            "Get the ledger out of the district."
        );
        assert_eq!(agency[0]["state"], "active");
    }

    /// **`/self/beliefs` returns the character's beliefs.**
    #[tokio::test]
    async fn self_beliefs_returns_the_belief_layer() {
        let npc = 43;
        let (rt, tokens) = daemon(Some(vec![payload(
            npc,
            vec![],
            vec![belief("b0", "Hess burned the granary.")],
        )]));
        let body = get(&rt, &tokens, npc, "/self/beliefs").await;
        let beliefs = body["beliefs"].as_array().expect("a beliefs array");
        assert_eq!(beliefs.len(), 1);
        assert_eq!(beliefs[0]["belief_id"], "b0");
        assert_eq!(beliefs[0]["statement"], "Hess burned the granary.");
    }

    /// **A character with nothing written reads as empty**, not as an error.
    #[tokio::test]
    async fn a_character_with_no_agency_reads_empty() {
        let npc = 44;
        let (rt, tokens) = daemon(Some(vec![payload(npc, vec![], vec![])]));
        let plan = get(&rt, &tokens, npc, "/self/plan").await;
        assert_eq!(plan["agency"].as_array().unwrap().len(), 0);
        let beliefs = get(&rt, &tokens, npc, "/self/beliefs").await;
        assert_eq!(beliefs["beliefs"].as_array().unwrap().len(), 0);
        // Memory has no layer on the record, so it is always empty here.
        let memory = get(&rt, &tokens, npc, "/self/memory").await;
        assert_eq!(memory["memory"].as_array().unwrap().len(), 0);
    }

    /// **The handle-not-installed path degrades, it does not panic.** A daemon
    /// stood up with no cast (a bare test runtime) still answers `/self/plan`
    /// with an empty layer — the read is always safe to make (§7.2).
    #[tokio::test]
    async fn a_missing_cast_degrades_to_empty() {
        let (rt, tokens) = daemon(None);
        let body = get(&rt, &tokens, 99, "/self/plan").await;
        assert_eq!(body["agency"].as_array().expect("still an array").len(), 0);
    }
}
