//! `http://local/reshape/:world` — runtime topology mutation, the operator and
//! embedder surface (effector design Appendix F).
//!
//! This is the door onto [`World::reshape`](npc_map::World::reshape): adding a
//! level, drowning a room, opening a gate that was not there — reshaping the
//! *walkable* map of a running world. It is deliberately **not** the Makers'
//! `/map` cartography station, which authors world-map *lore* (a `Kind::Place`
//! record item, a coastline the Makers draw as content); reshaping the rooms
//! bodies actually move through is an operator's or an embedder's act, not a
//! character's, so it lives on its own prefix and behind its own gate.
//!
//! # Direct scope only
//!
//! The device surface resolves a caller to a body and a [`Scope`]. An in-fiction
//! call carries [`Scope::AsNpc`]; this route requires [`Scope::Direct`] — the
//! operator/embedder scope (§8.3) — and refuses an as-npc token with `403`. A
//! Maker cannot reshape the vault it is standing in by reaching for its own
//! device, because a Maker's token is never `Direct`
//! ([`Tokens::as_npc`](crate::effector::token::Tokens::as_npc) forces it away).
//!
//! # The edit is the body, the world is the path
//!
//! `Direct` scope addresses anything regardless of standpoint, so the world to
//! reshape is named in the URL rather than resolved from a bound body, and the
//! [`MapEdit`] to apply is the JSON body — the adjacently tagged wire form
//! (`{"op":"add_node","with":{…}}`). A malformed edit comes back as a
//! prescriptive `400` the caller corrects against (§12), never a crash.
//!
//! # The response tells the truth about durability
//!
//! A reshape is committed to the running world first and written back to the
//! authored YAML second ([`Hosted::reshape`]). The response says which happened:
//! `durable: "written"` when the change is on disk, `"ephemeral"` for a world
//! with no authored file (it holds for the run only), and `"failed"` with the
//! reason when the change is live but the write did not land — so the operator
//! learns the change took but is not yet safe from a restart, rather than being
//! told a comfortable half-truth.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Json, Router};
use serde_json::{json, Value};

use npc_map::MapEdit;

use crate::api::err;
use crate::effector::auth::DeviceCaller;
use crate::effector::router::Local;
use crate::effector::token::Scope;
use crate::world::Durability;

/// The `/reshape` sub-router: one route, `POST /reshape/:world`.
///
/// Returned without state so the outer router supplies its [`Local`] and the
/// device-auth layer, exactly as the other effector sub-routers are mounted. It
/// is not advertised in the near-you index: an operator knows the endpoint, and
/// a character must never be shown a way to reshape the world it lives in.
pub fn routes() -> Router<Local> {
    Router::new().route("/:world", post(reshape_world))
}

/// `POST /reshape/:world` — reshape a hosted world's walkable map.
async fn reshape_world(
    State(local): State<Local>,
    Extension(caller): Extension<DeviceCaller>,
    Path(world): Path<String>,
    payload: Option<Json<Value>>,
) -> Response {
    // Direct scope only — reshaping the world is an operator/embedder act.
    if caller.scope != Scope::Direct {
        return err(
            StatusCode::FORBIDDEN,
            "forbidden",
            "reshaping the world needs a direct-scope operator token, not a character's own",
        );
    }
    let Some(runtime) = local.runtime() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_world",
            "the world is not running",
        );
    };
    let Some(hosted) = runtime.hosted.get(&world) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_world",
            &format!("no world `{world}` is hosted here"),
        );
    };
    // The edit is the body, parsed prescriptively so a malformed one teaches
    // rather than crashes (§12).
    let Some(Json(raw)) = payload else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_edit",
            "a reshape needs a map edit as the JSON body — e.g. \
             {\"op\":\"add_node\",\"with\":{\"area\":\"vault-casting\",\"node\":{…}}}",
        );
    };
    let edit: MapEdit = match serde_json::from_value(raw) {
        Ok(edit) => edit,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                "bad_edit",
                &format!("that is not a map edit: {e}"),
            )
        }
    };

    match hosted.reshape(&edit) {
        Ok(reshaped) => {
            let (durable, detail) = match reshaped.durability {
                Durability::Written => ("written", None),
                Durability::Ephemeral => (
                    "ephemeral",
                    Some("this world has no authored map on disk, so the change holds for this run only".to_string()),
                ),
                Durability::Failed(reason) => ("failed", Some(reason)),
            };
            let mut body = json!({
                "ok": true,
                "relocated": reshaped.relocated,
                "durable": durable,
            });
            if let Some(detail) = detail {
                body["detail"] = Value::String(detail);
            }
            Json(body).into_response()
        }
        // The world's own words on why it would not take the edit.
        Err(reason) => err(StatusCode::CONFLICT, "refused", &reason),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
    use axum::http::{Request, StatusCode};
    use npc_map::world::{Where, World};
    use npc_map::MapSet;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use crate::effector::router::{router, Local};
    use crate::effector::token::{Scope, Tokens};
    use crate::engine::runtime::Runtime;
    use crate::mind::Mind;
    use crate::world::Hosted;

    const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
    const WORLD: &str = "creators-vault";

    fn tmp() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "npcd-reshape-ut-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// A runtime holding the vault **in memory** (`Hosted::of`, no authored
    /// directory) so a reshape is ephemeral and never writes to the repo's map
    /// files — the disk-writeback path is pinned in `world::mapstore`'s own tests
    /// against a temp directory.
    fn daemon() -> (Arc<Runtime>, Arc<Tokens>) {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir());
        let map = MapSet::load_dir(ROOMS).expect("the vault loads");
        rt.hosted.keep(Arc::new(Hosted::of(WORLD, World::new(map))));
        let tokens = Arc::new(Tokens::load(tmp()).expect("a fresh token store"));
        rt.set_tokens(tokens.clone());
        (rt, tokens)
    }

    async fn send(
        rt: &Arc<Runtime>,
        tokens: &Arc<Tokens>,
        token: &str,
        path: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let app = router(Local::new(tokens.clone(), rt));
        let builder = Request::builder()
            .method("POST")
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

    fn add_annex() -> Value {
        json!({
            "op": "add_node",
            "with": {
                "area": "vault-casting",
                "node": {
                    "id": "annex",
                    "kind": "social",
                    "name": "the annex",
                    "off": ["green-room"]
                }
            }
        })
    }

    /// **A direct-scope operator reshapes the world.** The edit lands, nobody is
    /// displaced, and — this world being in-memory — it reports itself ephemeral.
    #[tokio::test]
    async fn a_direct_operator_can_reshape() {
        let (rt, tokens) = daemon();
        let token = tokens.mint(900, Scope::Direct).expect("a direct token");
        let (status, out) = send(
            &rt,
            &tokens,
            &token,
            "/reshape/creators-vault",
            Some(add_annex()),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{out}");
        assert_eq!(out["ok"], json!(true), "{out}");
        assert_eq!(out["relocated"], json!([]), "{out}");
        assert_eq!(out["durable"], json!("ephemeral"), "{out}");
        // And the walkable map really changed — the annex is now a node.
        let there = rt
            .hosted
            .get(WORLD)
            .unwrap()
            .read(|w| w.node(&Where::new("vault-casting", "annex")).is_some());
        assert!(there, "the reshape did not reach the live map");
    }

    /// **An as-npc token is refused.** A character cannot reshape the world it
    /// stands in through its own device — reshaping is a direct-scope act.
    #[tokio::test]
    async fn an_as_npc_token_is_forbidden() {
        let (rt, tokens) = daemon();
        let token = tokens.mint(901, Scope::AsNpc).expect("an as-npc token");
        let (status, out) = send(
            &rt,
            &tokens,
            &token,
            "/reshape/creators-vault",
            Some(add_annex()),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{out}");
        assert_eq!(out["error"], json!("forbidden"), "{out}");
    }

    /// **An unknown world is a `404`.**
    #[tokio::test]
    async fn an_unknown_world_is_not_found() {
        let (rt, tokens) = daemon();
        let token = tokens.mint(902, Scope::Direct).unwrap();
        let (status, out) = send(
            &rt,
            &tokens,
            &token,
            "/reshape/no-such-world",
            Some(add_annex()),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{out}");
        assert_eq!(out["error"], json!("no_such_world"), "{out}");
    }

    /// **A malformed edit is a prescriptive `400`, not a crash.**
    #[tokio::test]
    async fn a_malformed_edit_is_a_bad_request() {
        let (rt, tokens) = daemon();
        let token = tokens.mint(903, Scope::Direct).unwrap();
        let (status, out) = send(
            &rt,
            &tokens,
            &token,
            "/reshape/creators-vault",
            Some(json!({ "op": "nonsense" })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{out}");
        assert_eq!(out["error"], json!("bad_edit"), "{out}");
    }

    /// **An edit the world will not take is a `409` in its own words.** Adding a
    /// room that opens off nowhere cannot validate, so the reshape is refused and
    /// nothing changes.
    #[tokio::test]
    async fn an_impossible_edit_is_refused() {
        let (rt, tokens) = daemon();
        let token = tokens.mint(904, Scope::Direct).unwrap();
        let edit = json!({
            "op": "add_node",
            "with": {
                "area": "vault-casting",
                "node": { "id": "annex", "kind": "social", "name": "the annex", "off": ["nowhere"] }
            }
        });
        let (status, out) = send(&rt, &tokens, &token, "/reshape/creators-vault", Some(edit)).await;
        assert_eq!(status, StatusCode::CONFLICT, "{out}");
        assert_eq!(out["error"], json!("refused"), "{out}");
    }
}
