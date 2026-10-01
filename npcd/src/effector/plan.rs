//! `http://local/plan/<id>` and `http://local/orders/<id>` — the projected
//! store (effector design §9.2, Appendix C.11/C.21, D.6 #2).
//!
//! These are the one effector surface that does **not** change the world and is
//! **not** perceived over ticks. A `plan_*`/`orders_*` call writes the
//! character's own *maintained state* — its plan and the orders it holds — into
//! the substrate **agency** layer, and that layer projects straight into the
//! next turn's system prompt (the `agency` collection in `projection.yaml`, and
//! [`crate::engine::persona`] rendering the first active strategy as the
//! character's intent). The character reads its plan each turn; it does not
//! re-discover it from an event feed (§9.2).
//!
//! # The bridge, not `body::perform`
//!
//! Every other station verb runs through the synchronous world dispatch
//! ([`crate::engine::body::perform`], `&Hosted`-only), which writes `Sim`/`World`
//! in RAM. That is wrong for these two namespaces: §9.2 requires the agency
//! *plane*, which is the substrate — persisted, redo-logged, projected. So these
//! verbs run at the **async** layer instead, writing the shared cast through
//! [`Npcs::put_strategy_self`](crate::npcs::Npcs::put_strategy_self): owner-blind,
//! because the effector is the character's own body and carries no operator owner
//! (Appendix E "The bridge"). The write supersedes the record and projects on the
//! next turn with no extra wiring.
//!
//! The `:id` is the **strategy id**, not a map instance id — the character's own
//! goal keyed by name, so a write is idempotent (a second `scope` of the same id
//! revises it). No reach check: a character's plan is its own, reachable wherever
//! it stands (§7.2).
//!
//! # The verb → strategy mapping
//!
//! Pragmatic and documented (effector design §9.2 leaves the exact shape to the
//! implementation), translating each verb's body into a
//! [`Npcs::put_strategy`](crate::npcs::Npcs::put_strategy) write:
//!
//! | verb (namespace) | target | write |
//! |---|---|---|
//! | `scope` (plan) | `:id` | create/replace an active strategy, statement = `what` (+ `in`/`out` as scope notes) |
//! | `set` (orders) | `:id` | create/replace an active strategy, statement = `what` (+ `for`) |
//! | `take` (orders) | `:id` | create/replace an active strategy, statement = `what` |
//! | `hand_to` (orders) | `:id` | create/replace an active strategy, statement = `what` — for `to` |
//! | `order`/`reorder` (plan) | `:id` | create/replace an active strategy, statement = the ordered `pieces` |
//! | `break_down` (plan) | a new child of `:id` | a child strategy (`parent_id = :id`), statement = `what` |
//! | `report_done` (orders) | `:id` | move the strategy to `finished` |
//! | `give_back` (orders) | `:id` | move the strategy to `abandoned` |
//!
//! `give_back` maps to `abandoned` rather than the design's informal "dormant":
//! the agency layer's states are exactly `active`/`finished`/`abandoned`
//! ([`crate::npcs::Npcs::put_strategy`]), and a state off that list is not
//! writable. `report_done`/`give_back` carry the strategy's existing statement
//! forward, so a verb that names a strategy the character never had is refused
//! (an empty statement, `invalid_field`) — you cannot report done something you
//! never took.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde_json::{json, Map, Value};

use crate::api::{err, now_ms};
use crate::effector::auth::DeviceCaller;
use crate::effector::router::Local;
use crate::npcs::{self, NpcError};

/// The sub-router for a projected-agency namespace: read the layer, or write it.
///
/// The *same* router is nested under both `/plan` and `/orders`
/// ([`crate::effector::router::router`]); the namespace only picks which verbs a
/// character reaches for, and each verb resolves to the same agency write.
pub fn routes() -> Router<Local> {
    Router::new()
        .route("/:id", get(read))
        .route("/:id/:verb", post(write))
}

/// `GET /<ns>/:id` — the character's agency layer as a projected read.
///
/// The read is the caller's own whole layer, not just `:id`: the plan is a tree,
/// and a character reads all of it. Degrades to an empty layer if the cast is not
/// installed, the same as the `/self` reads.
async fn read(State(local): State<Local>, Extension(caller): Extension<DeviceCaller>) -> Response {
    let Some(runtime) = local.runtime() else {
        return Json(json!({ "agency": [] })).into_response();
    };
    let Some(cast) = runtime.npcs() else {
        return Json(json!({ "agency": [] })).into_response();
    };
    let rendered = {
        let cast = cast.read().await;
        cast.payload(caller.npc_id).map(npcs::agency_wire)
    };
    Json(rendered.unwrap_or_else(|| json!({ "agency": [] }))).into_response()
}

/// `POST /<ns>/:id/:verb` — write the character's own agency plane (§9.2).
async fn write(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path((id, verb)): Path<(String, String)>,
    payload: Option<Json<Value>>,
) -> Response {
    let Some(runtime) = local.runtime() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_world",
            "the world is not running",
        );
    };
    let Some(cast) = runtime.npcs() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_cast",
            "the cast is not installed, so there is no plan to write",
        );
    };
    let body = match payload {
        Some(Json(Value::Object(fields))) => fields,
        _ => Map::new(),
    };

    // The write acts as the character's own body, so it takes the cast's write
    // lock — never the world lock — and holds it across the read-then-write so a
    // `break_down`'s child id is computed against the same layer it is appended
    // to. No `.await` under this guard reaches the world.
    let mut cast = cast.write().await;
    let now = now_ms();

    // Each verb hands back the write's result *and* the one-line ack the
    // character reads this turn — the ack fits the verb it names (setting a goal
    // down reads differently from finishing one or handing it back), and a
    // `break_down` can only name the child it just minted from inside its arm.
    let (outcome, ack) = match verb.as_str() {
        // Create or replace the named strategy, active.
        "scope" => {
            let statement = with_notes(text(&body, "what"), &[("in", &body), ("out", &body)]);
            (
                cast.put_strategy_self(caller.npc_id, &id, &active(statement), now),
                format!("You set down `{id}`."),
            )
        }
        "set" => {
            let statement = with_notes(text(&body, "what"), &[("for", &body)]);
            (
                cast.put_strategy_self(caller.npc_id, &id, &active(statement), now),
                format!("You set down `{id}`."),
            )
        }
        "take" => {
            let statement = text(&body, "what");
            (
                cast.put_strategy_self(caller.npc_id, &id, &active(statement), now),
                format!("You set down `{id}`."),
            )
        }
        "hand_to" => {
            let what = text(&body, "what");
            let to = text(&body, "to");
            let statement = match (what.is_empty(), to.is_empty()) {
                (false, false) => format!("{what} — for {to}"),
                (false, true) => what,
                (true, false) => to,
                (true, true) => what,
            };
            (
                cast.put_strategy_self(caller.npc_id, &id, &active(statement), now),
                format!("You set down `{id}`."),
            )
        }
        "order" | "reorder" => {
            let statement = pieces(&body);
            (
                cast.put_strategy_self(caller.npc_id, &id, &active(statement), now),
                format!("You set down `{id}`."),
            )
        }
        // A child of `:id`: a sub-goal in the plan tree, keyed under a fresh id.
        "break_down" => {
            let child = child_id(&cast, caller.npc_id, &id);
            let statement = text(&body, "what");
            (
                cast.put_strategy_self(
                    caller.npc_id,
                    &child,
                    &json!({ "statement": statement, "parent_id": id }),
                    now,
                ),
                format!("You break `{id}` down into `{child}`."),
            )
        }
        // Move the strategy's state on; its statement carries forward.
        "report_done" => (
            cast.put_strategy_self(caller.npc_id, &id, &state("finished"), now),
            format!("You mark `{id}` done."),
        ),
        "give_back" => (
            cast.put_strategy_self(caller.npc_id, &id, &state("abandoned"), now),
            format!("You hand `{id}` back."),
        ),
        other => {
            return err(
                StatusCode::NOT_FOUND,
                "unknown_verb",
                &format!("`{other}` is not something you can do to a plan or an order"),
            );
        }
    };
    drop(cast);

    match outcome {
        // The line the character reads back this turn — the world took the write,
        // and the effect is that the plan now projects. Mirrors the `enact_route`
        // `Did` shape so the in-fiction `invoke` reads `ok` the same way.
        Ok(_) => (StatusCode::OK, Json(json!({ "ok": true, "detail": ack }))).into_response(),
        Err(e) => write_err(e),
    }
}

/// A put-strategy body that creates or replaces an **active** strategy with this
/// statement.
fn active(statement: String) -> Value {
    json!({ "statement": statement, "state": "active" })
}

/// A put-strategy body that only moves an existing strategy's state.
fn state(s: &str) -> Value {
    json!({ "state": s })
}

/// A string field off the verb body, trimmed; empty when absent.
fn text(body: &Map<String, Value>, key: &str) -> String {
    body.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Append `key: value` scope notes to a statement, for the optional qualifiers
/// (`in`/`out` on a scope, `for` on an order) that narrow what a goal covers.
fn with_notes(mut statement: String, notes: &[(&str, &Map<String, Value>)]) -> String {
    let mut clauses = Vec::new();
    for (key, body) in notes {
        let v = text(body, key);
        if !v.is_empty() {
            clauses.push(format!("{key}: {v}"));
        }
    }
    if !clauses.is_empty() {
        if statement.is_empty() {
            statement = clauses.join("; ");
        } else {
            statement = format!("{statement} ({})", clauses.join("; "));
        }
    }
    statement
}

/// The `pieces` array of an `order`/`reorder`, joined into one ordered statement.
fn pieces(body: &Map<String, Value>) -> String {
    body.get("pieces")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default()
}

/// A fresh child id for a `break_down`, `<parent>-<n>` where `n` starts at how
/// many children the parent already has and advances past any id that is
/// already taken by a *different* strategy — a directly-authored `widget-0`
/// with no parent, say. Without that check the guessed `<parent>-<n>` could
/// name an unrelated existing strategy, and `put_strategy_self`'s
/// create-or-replace would silently overwrite and reparent it. Computed under
/// the write lock, so two break-downs in a row do not collide with each other
/// either.
fn child_id(cast: &npcs::Npcs, npc_id: u64, parent: &str) -> String {
    let Some(payload) = cast.payload(npc_id) else {
        return format!("{parent}-0");
    };
    let taken: std::collections::HashSet<&str> = payload
        .agency
        .iter()
        .map(|s| s.strategy_id.as_str())
        .collect();
    let mut n = payload
        .agency
        .iter()
        .filter(|s| s.parent_id.as_deref() == Some(parent))
        .count();
    let mut candidate = format!("{parent}-{n}");
    while taken.contains(candidate.as_str()) {
        n += 1;
        candidate = format!("{parent}-{n}");
    }
    candidate
}

/// Map a cast write error to the route's response, in the estate error shape.
///
/// `NotFound` is the character reading as absent (a `404`); `Invalid` is a
/// refusal the world would not take — an empty statement, a bad parent — mapped
/// to `409` like every other effector refusal (§12); a persist failure is a
/// `500`.
fn write_err(e: NpcError) -> Response {
    match e {
        NpcError::NotFound => err(StatusCode::NOT_FOUND, "npc_not_found", "no such character"),
        NpcError::Invalid(field) => err(
            StatusCode::CONFLICT,
            "refused",
            &format!("`{field}` is missing or out of range"),
        ),
        NpcError::Persist(detail) => {
            tracing::error!(error = %detail, "plan/orders write failed");
            err(StatusCode::INTERNAL_SERVER_ERROR, "write_failed", &detail)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
    use axum::http::{Request, StatusCode};
    use candle_conversation::persistence::record::{Modulation, NpcPayload};
    use serde_json::{json, Value};
    use tokio::sync::RwLock as AsyncRwLock;
    use tower::ServiceExt;

    use crate::effector::router::{router, Local};
    use crate::effector::token::{Scope, Tokens};
    use crate::engine::persona;
    use crate::engine::runtime::Runtime;
    use crate::mind::Mind;
    use crate::npcs::Npcs;

    const OWNER: &str = "u_1a2b3c4d";

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-plan-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn payload(npc_id: u64) -> NpcPayload {
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
            beliefs: Vec::new(),
            relationships: Vec::new(),
            agency: Vec::new(),
            modulation: Modulation::default(),
        }
    }

    /// A runtime with a token store and a seeded, cast-installed character — the
    /// cast Arc is handed back so a test can read the record after a write.
    fn daemon(npc_id: u64) -> (Arc<Runtime>, Arc<Tokens>, Arc<AsyncRwLock<Npcs>>) {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
        let tokens = Arc::new(Tokens::load(tmp()).expect("a fresh token store"));
        rt.set_tokens(tokens.clone());
        let mut npcs = Npcs::load(&tmp()).expect("a fresh cast");
        npcs.import(vec![payload(npc_id)]).expect("seeded");
        let cast = Arc::new(AsyncRwLock::new(npcs));
        rt.set_npcs(cast.clone());
        (rt, tokens, cast)
    }

    async fn post(
        rt: &Arc<Runtime>,
        tokens: &Arc<Tokens>,
        npc_id: u64,
        path: &str,
        body: Value,
    ) -> (StatusCode, Value) {
        let token = tokens.mint(npc_id, Scope::AsNpc).expect("minted");
        let app = router(Local::new(tokens.clone(), rt));
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
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

    /// **The load-bearing outcome (§9.2): a `plan_scope` writes the agency layer
    /// and therefore projects.** The `POST` lands, the character's own
    /// `payload().agency` then holds the strategy, active — and
    /// [`persona::of`]/`intent` renders it, which is the exact projection the
    /// character reads in its next turn's system prompt.
    #[tokio::test]
    async fn a_plan_scope_writes_the_agency_layer_and_projects() {
        let npc = 7;
        let (rt, tokens, cast) = daemon(npc);

        let (status, ack) = post(
            &rt,
            &tokens,
            npc,
            "/plan/hold-the-district/scope",
            json!({ "what": "Get the ledger out of the district." }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{ack}");
        assert_eq!(ack["ok"], json!(true), "{ack}");

        // It is in the agency layer, active — a durable superseding record.
        let guard = cast.read().await;
        let held = guard.payload(npc).expect("still there");
        let strat = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "hold-the-district")
            .expect("the strategy the scope wrote");
        assert_eq!(strat.statement, "Get the ledger out of the district.");
        assert_eq!(strat.state, "active");

        // And it projects: the persona render surfaces it as the intent, which is
        // what the dynamic system prompt carries every turn.
        let rendered = persona::of(held, "", "");
        assert_eq!(
            rendered.intent.as_deref(),
            Some("Get the ledger out of the district."),
            "the plan did not project into the persona as the intent"
        );
    }

    /// **`report_done` moves the strategy on**, and it is the same record — the
    /// projection stops surfacing a finished goal as the active intent.
    #[tokio::test]
    async fn report_done_finishes_the_strategy_and_it_leaves_the_intent() {
        let npc = 8;
        let (rt, tokens, cast) = daemon(npc);

        post(
            &rt,
            &tokens,
            npc,
            "/orders/take-the-toll/set",
            json!({ "what": "Hold the toll gate." }),
        )
        .await;
        let (status, ack) = post(
            &rt,
            &tokens,
            npc,
            "/orders/take-the-toll/report_done",
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // The ack fits the verb — marking done, not "setting down".
        assert_eq!(
            ack["detail"],
            json!("You mark `take-the-toll` done."),
            "{ack}"
        );

        let guard = cast.read().await;
        let held = guard.payload(npc).expect("still there");
        let strat = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "take-the-toll")
            .expect("the order");
        assert_eq!(strat.state, "finished");
        // A finished goal is no longer the intent (`persona::intent` takes the
        // first *active* one).
        assert_eq!(persona::of(held, "", "").intent, None);
    }

    /// **`give_back` hands the order back**, and its ack fits the verb rather
    /// than reading as "setting down".
    #[tokio::test]
    async fn give_back_abandons_the_order_with_its_own_ack() {
        let npc = 11;
        let (rt, tokens, cast) = daemon(npc);

        post(
            &rt,
            &tokens,
            npc,
            "/orders/mind-the-gate/set",
            json!({ "what": "Mind the gate." }),
        )
        .await;
        let (status, ack) = post(
            &rt,
            &tokens,
            npc,
            "/orders/mind-the-gate/give_back",
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{ack}");
        assert_eq!(
            ack["detail"],
            json!("You hand `mind-the-gate` back."),
            "{ack}"
        );

        let guard = cast.read().await;
        let held = guard.payload(npc).expect("still there");
        let strat = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "mind-the-gate")
            .expect("the order");
        assert_eq!(strat.state, "abandoned");
    }

    /// **`break_down` adds a child under the parent** — the sub-goal tree §9.2's
    /// plan is (`parent_id`).
    #[tokio::test]
    async fn break_down_adds_a_child_of_the_parent() {
        let npc = 9;
        let (rt, tokens, cast) = daemon(npc);

        post(
            &rt,
            &tokens,
            npc,
            "/plan/win-the-siege/scope",
            json!({ "what": "Win the siege." }),
        )
        .await;
        let (status, ack) = post(
            &rt,
            &tokens,
            npc,
            "/plan/win-the-siege/break_down",
            json!({ "what": "Cut the supply road first." }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        // The ack names the child it just minted, not the parent.
        assert_eq!(
            ack["detail"],
            json!("You break `win-the-siege` down into `win-the-siege-0`."),
            "{ack}"
        );

        let guard = cast.read().await;
        let held = guard.payload(npc).expect("still there");
        let child = held
            .agency
            .iter()
            .find(|s| s.parent_id.as_deref() == Some("win-the-siege"))
            .expect("a child of the parent");
        assert_eq!(child.statement, "Cut the supply road first.");
        assert_eq!(child.strategy_id, "win-the-siege-0");
    }

    /// **`break_down` never mints an id that collides with an unrelated
    /// strategy.** `widget-0` already exists as its own top-level goal before
    /// `widget` is broken down for the first time; the naive `<parent>-<n>`
    /// guess would silently overwrite and reparent it. The child must instead
    /// land on the next id that is actually free.
    #[tokio::test]
    async fn break_down_never_collides_with_an_unrelated_strategy() {
        let npc = 12;
        let (rt, tokens, cast) = daemon(npc);

        post(
            &rt,
            &tokens,
            npc,
            "/plan/widget/scope",
            json!({ "what": "Ship the widget." }),
        )
        .await;
        post(
            &rt,
            &tokens,
            npc,
            "/plan/widget-0/scope",
            json!({ "what": "An unrelated top-level goal." }),
        )
        .await;

        let (status, ack) = post(
            &rt,
            &tokens,
            npc,
            "/plan/widget/break_down",
            json!({ "what": "Get parts." }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{ack}");
        // The child landed on the next free id, not on `widget-0`.
        assert_eq!(
            ack["detail"],
            json!("You break `widget` down into `widget-1`."),
            "{ack}"
        );

        let guard = cast.read().await;
        let held = guard.payload(npc).expect("still there");
        // The unrelated goal is untouched — still its own statement, still no
        // parent.
        let unrelated = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "widget-0")
            .expect("the unrelated goal survives");
        assert_eq!(unrelated.statement, "An unrelated top-level goal.");
        assert_eq!(unrelated.parent_id, None);
        // And the real child exists, parented correctly.
        let child = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "widget-1")
            .expect("the child landed on the free id");
        assert_eq!(child.parent_id.as_deref(), Some("widget"));
        assert_eq!(child.statement, "Get parts.");
    }

    /// **`hand_to` keeps the recipient even when `what` is not repeated.** A
    /// caller that only names who it is handing back to must not have that
    /// silently dropped in favour of an empty statement.
    #[tokio::test]
    async fn hand_to_keeps_the_recipient_when_what_is_absent() {
        let npc = 13;
        let (rt, tokens, cast) = daemon(npc);

        post(
            &rt,
            &tokens,
            npc,
            "/orders/watch-the-gate/set",
            json!({ "what": "Watch the gate." }),
        )
        .await;
        let (status, ack) = post(
            &rt,
            &tokens,
            npc,
            "/orders/watch-the-gate/hand_to",
            json!({ "to": "Bob" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{ack}");

        let guard = cast.read().await;
        let held = guard.payload(npc).expect("still there");
        let strat = held
            .agency
            .iter()
            .find(|s| s.strategy_id == "watch-the-gate")
            .expect("the order");
        assert_eq!(strat.statement, "Bob", "the recipient was dropped");
    }

    /// **An unknown strategy cannot be reported done.** `report_done` carries the
    /// existing statement forward, so naming one the character never took is an
    /// empty statement — refused `409`, not a phantom finished goal.
    #[tokio::test]
    async fn reporting_done_a_strategy_you_never_had_is_refused() {
        let npc = 10;
        let (rt, tokens, _cast) = daemon(npc);
        let (status, refused) = post(
            &rt,
            &tokens,
            npc,
            "/orders/never-took-this/report_done",
            json!({}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{refused}");
        assert_eq!(refused["error"], json!("refused"));
    }
}
