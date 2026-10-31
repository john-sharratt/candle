//! `POST /v1/me/discord/link`, `GET /v1/me/discord`, `DELETE /v1/me/discord`
//! — linking the caller's own Discord account (`docs/npcd_discord.md` §3.2).
//!
//! The caller's own, like the rest of `/v1/me`: there is no id in these
//! paths to belong to anybody else.
//!
//! - `POST …/link` answers a fresh code, when it stops being good, and the
//!   names of the bots it can be sent to — every connected bot of a
//!   character the caller may message.
//! - `GET` says whether a Discord account is linked, and which.
//! - `DELETE` unlinks it.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::accounts::{key_of, user_id_of, DiscordUser};
use crate::api::{account_holder, err, now_ms, Authored};
use crate::discord::link::{code, linking};
use crate::discord::store;

/// `POST /v1/me/discord/link`
pub async fn post_link(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let now = now_ms();
    if let Err(e) = s.accounts.write().await.upsert(&id, now) {
        tracing::error!(error = %e, "discord: the account was not written");
        return failed();
    }
    let shared = s.npcs.substrate();
    let (code, expires_ms) = match code::mint(&shared, &key_of(&id), now) {
        Ok(minted) => minted,
        Err(e) => {
            tracing::error!(error = %e, "discord: a link code was not written");
            return failed();
        }
    };
    let mut bots: Vec<String> = s
        .npcs
        .view()
        .owned_by(&user_id_of(&id))
        .into_iter()
        .filter_map(|npc| store::get(&shared, npc).map(|bot| bot.username))
        .collect();
    bots.sort();
    Json(json!({ "code": code, "expires_ms": expires_ms, "bots": bots })).into_response()
}

/// `GET /v1/me/discord`
pub async fn get_link(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    Json(linked(
        s.accounts.read().await.discord(&key_of(&id)).as_ref(),
    ))
    .into_response()
}

/// `DELETE /v1/me/discord`
pub async fn delete_link(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    let id = match account_holder(&s, &headers) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let accounts = s.accounts.read().await;
    match linking::unlink(&accounts, &s.npcs.substrate(), &key_of(&id)) {
        Ok(Some(user)) => {
            tracing::info!(user = %user.id, "discord: an account was unlinked");
            Json(linked(None)).into_response()
        }
        Ok(None) => Json(linked(None)).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "discord: the link was not removed");
            failed()
        }
    }
}

/// The wire shape of a link, or of there being none.
fn linked(user: Option<&DiscordUser>) -> Value {
    match user {
        None => json!({ "linked": false }),
        Some(user) => json!({
            "linked": true,
            "username": user.username,
            "linked_ms": user.linked_ms,
        }),
    }
}

fn failed() -> Response {
    err(
        StatusCode::INTERNAL_SERVER_ERROR,
        "discord_write_failed",
        "that was not kept; try again",
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use axum::Router;
    use tempfile::TempDir;
    use tower::ServiceExt;
    use web::auth::session::Identity;

    use super::*;
    use crate::accounts::Accounts;
    use crate::api::router;
    use crate::collections::Libraries;
    use crate::discord::link::index;
    use crate::discord::store::Bot;
    use crate::images::Images;
    use crate::mind::Mind;
    use crate::npcs::{self, Npcs};
    use crate::projection::Source;
    use crate::registry::Registry;

    const OWNER: &str = "google-1";

    fn identity(sub: &str) -> Identity {
        Identity {
            provider: "google".into(),
            sub: sub.into(),
            email: "wren@example.com".into(),
            name: "Wren S".into(),
            picture: String::new(),
            exp: 9_999_999_999,
        }
    }

    /// npcd with no engine, and [`OWNER`]'s characters Zen, with a bot, and
    /// Pax, without one.
    fn state(dir: &TempDir) -> Arc<Authored> {
        let base = dir.path();
        let npcs = Npcs::load(base).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &base.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &base.join("images")).unwrap();
        let authored = Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            accounts,
            npcs,
            serde_yaml::from_str(&format!("people:\n  - sub: {OWNER}\n    role: user\n")).unwrap(),
            Libraries::load(&Source::resolve(None).unwrap()),
            Mind::new(None),
            images,
            BTreeMap::new(),
        );
        let owner = user_id_of(&identity(OWNER));
        for name in ["Zen", "Pax"] {
            let made = authored
                .npcs
                .create(
                    &identity(OWNER),
                    &owner,
                    &json!({ "name": name, "world_id": "life", "personality_id": "zen" }),
                    1,
                )
                .unwrap();
            if name == "Zen" {
                let npc = npcs::npc_id_of_wire(made["npc_id"].as_str().unwrap()).unwrap();
                store::put(
                    &authored.npcs.substrate(),
                    npc,
                    &Bot {
                        application_id: "111".into(),
                        user_id: "222".into(),
                        username: "Zen".into(),
                        token: "c2VhbGVk".into(),
                        connected_ms: 1,
                    },
                )
                .unwrap();
            }
        }
        Arc::new(authored)
    }

    fn request(method: &str, path: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header("x-tokera-user", OWNER)
            .header("x-tokera-provider", "google")
            .header("x-tokera-email", "wren@example.com")
            .header("x-tokera-name", "Wren S")
            .body(Body::empty())
            .unwrap()
    }

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    /// **A code is six digits, good for ten minutes, sent to the caller's
    /// characters' bots** — only the ones that have one.
    #[tokio::test]
    async fn asking_for_a_code_answers_it_and_the_bots_to_send_it_to() {
        let dir = TempDir::new().unwrap();
        let st = state(&dir);
        let before = now_ms();
        let (s, v) = call(router(st.clone()), request("POST", "/v1/me/discord/link")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let code = v["code"].as_str().unwrap().to_string();
        assert!(code::is_code(&code), "{code}");
        let expires_ms = v["expires_ms"].as_u64().unwrap();
        assert!(expires_ms >= before + code::VALID_MS, "{expires_ms}");
        assert_eq!(
            v,
            json!({ "code": code, "expires_ms": expires_ms, "bots": ["Zen"] })
        );
        assert_eq!(
            code::standing(&st.npcs.substrate(), &key_of(&identity(OWNER)), now_ms()),
            Some((code, expires_ms))
        );
    }

    /// **`GET` says whether a Discord account is linked; `DELETE` unlinks
    /// it**, both halves.
    #[tokio::test]
    async fn the_link_is_read_and_removed() {
        let dir = TempDir::new().unwrap();
        let st = state(&dir);
        let (s, v) = call(router(st.clone()), request("GET", "/v1/me/discord")).await;
        assert_eq!((s, v), (StatusCode::OK, json!({ "linked": false })));

        let key = key_of(&identity(OWNER));
        st.accounts
            .write()
            .await
            .upsert(&identity(OWNER), 1)
            .unwrap();
        let user = DiscordUser {
            id: "77".into(),
            username: "wren".into(),
            linked_ms: 1_800_000_000_000,
        };
        linking::link(
            &*st.accounts.read().await,
            &st.npcs.substrate(),
            &key,
            &user,
        )
        .unwrap();
        let (_, v) = call(router(st.clone()), request("GET", "/v1/me/discord")).await;
        assert_eq!(
            v,
            json!({ "linked": true, "username": "wren", "linked_ms": 1_800_000_000_000u64 })
        );

        let (s, v) = call(router(st.clone()), request("DELETE", "/v1/me/discord")).await;
        assert_eq!((s, v), (StatusCode::OK, json!({ "linked": false })));
        assert_eq!(index::account_of(&st.npcs.substrate(), "77"), None);
        let (_, v) = call(router(st.clone()), request("GET", "/v1/me/discord")).await;
        assert_eq!(v, json!({ "linked": false }));
    }

    #[tokio::test]
    async fn a_stranger_is_asked_to_sign_in() {
        let dir = TempDir::new().unwrap();
        let st = state(&dir);
        let anonymous = Request::builder()
            .method("POST")
            .uri("/v1/me/discord/link")
            .body(Body::empty())
            .unwrap();
        let (s, _) = call(router(st), anonymous).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
    }
}
