//! `GET`/`PUT`/`DELETE /v1/npc/:nid/discord` — a character's own bot.
//!
//! The owner's, like everything else about a character: `User` is the bar
//! and ownership the rest of the answer, checked here, with a character that
//! is somebody else's answered exactly as one that does not exist (§8.3).
//!
//! - `GET` says whether the character has a bot, and if so which, with the
//!   link that invites it to a server and its status (§3.3): `online` — with
//!   the world it is online in — `offline` or `refused`.
//! - `PUT {"token": …}` connects one: the token is checked with Discord
//!   (`GET /users/@me`, refusing a user's token, and `GET /applications/@me`),
//!   the bot is named as the character and pictured with its portrait if it
//!   has one (`PATCH /users/@me`, sending only what differs), and the token is
//!   sealed into the bot's record. Nothing is written unless all of that
//!   succeeded.
//! - `DELETE` removes the record. The bot itself stays the owner's, in the
//!   portal.
//!
//! No answer carries the token, sealed or not.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{err, now_ms, npc_err, owner_of, Authored};
use crate::discord::error::DiscordError;
use crate::discord::invite;
use crate::discord::profile;
use crate::discord::state::{self, Status};
use crate::discord::store::{self, Bot};
use crate::npcs;

/// What `PUT` takes: the bot token, as the portal's Reset Token shows it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectBody {
    token: String,
}

/// `GET /v1/npc/:nid/discord`
pub async fn get_discord(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (npc_id, _) = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let bot = store::get(&s.npcs.substrate(), npc_id);
    Json(connection(&s, bot.as_ref().map(|bot| (npc_id, bot))).await).into_response()
}

/// `PUT /v1/npc/:nid/discord`
pub async fn put_discord(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<ConnectBody>,
) -> Response {
    let (npc_id, record) = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let token = body.token.trim();
    if token.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "discord_token_missing",
            "paste the bot's token from the Developer Portal",
        );
    }

    let me = match s.discord.me(token).await {
        Ok(me) => me,
        Err(e) => return refused(npc_id, e),
    };
    let app = match s.discord.application(token).await {
        Ok(app) => app,
        Err(e) => return refused(npc_id, e),
    };

    // Only what differs is sent: Discord allows a bot few renames an hour, so
    // connecting the same bot again must not spend one on its own name.
    let name = record["name"].as_str().unwrap_or_default();
    let rename = profile::username(name).filter(|n| *n != me.username);
    let avatar = record["portrait"]["image_id"]
        .as_str()
        .and_then(|iid| match s.images.get(iid) {
            Ok((bytes, mime)) => Some(profile::avatar(&bytes, mime)),
            Err(e) => {
                tracing::warn!(npc = npc_id, error = %e, "discord: the portrait was not read; the bot keeps its avatar");
                None
            }
        });
    let username = match (rename.as_deref(), avatar.as_deref()) {
        (None, None) => me.username,
        (rename, avatar) => match s.discord.set_profile(token, rename, avatar).await {
            Ok(user) => user.username,
            Err(e) => return refused(npc_id, e),
        },
    };

    let bot = Bot {
        application_id: app.id,
        user_id: me.id,
        username,
        token: s.discord_tokens.seal(npc_id, token),
        connected_ms: now_ms(),
    };
    if let Err(e) = store::put(&s.npcs.substrate(), npc_id, &bot) {
        tracing::error!(npc = npc_id, error = %e, "discord: the bot was not written");
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "discord_write_failed",
            "the bot was checked but not kept; connect it again",
        );
    }
    tracing::info!(
        npc = npc_id,
        user = %bot.user_id,
        application = %bot.application_id,
        "discord: bot connected"
    );
    Json(connection(&s, Some((npc_id, &bot))).await).into_response()
}

/// `DELETE /v1/npc/:nid/discord`
pub async fn delete_discord(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
) -> Response {
    let (npc_id, _) = match owned(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let shared = s.npcs.substrate();
    if store::get(&shared, npc_id).is_some() {
        if let Err(e) = store::remove(&shared, npc_id) {
            tracing::error!(npc = npc_id, error = %e, "discord: the bot was not removed");
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "discord_write_failed",
                "the bot was not removed; try again",
            );
        }
        tracing::info!(npc = npc_id, "discord: bot disconnected");
    }
    Json(connection(&s, None).await).into_response()
}

/// The caller's character `nid`: its id and its record, or the 404 that
/// answers a character that is missing and one that is somebody else's alike.
async fn owned(
    s: &Arc<Authored>,
    headers: &HeaderMap,
    nid: &str,
) -> Result<(u64, Value), Box<Response>> {
    let (_, owner) = owner_of(s, headers).await?;
    let Some(npc_id) = npcs::npc_id_of_wire(nid) else {
        return Err(Box::new(err(
            StatusCode::NOT_FOUND,
            "npc_not_found",
            "no such character",
        )));
    };
    let record = s
        .npcs
        .view()
        .get(npc_id, &owner)
        .map_err(|e| Box::new(npc_err(e)))?;
    Ok((npc_id, record))
}

/// The wire shape of character `npc_id`'s bot, or of its having none: which
/// bot it is, and its status — with, when it is online, the name of the
/// world it is online in.
async fn connection(s: &Authored, bot: Option<(u64, &Bot)>) -> Value {
    let Some((npc_id, bot)) = bot else {
        return json!({ "connected": false });
    };
    let status = state::status(&s.npcs.substrate(), npc_id, bot);
    let mut wire = json!({
        "connected": true,
        "application_id": bot.application_id,
        "user_id": bot.user_id,
        "username": bot.username,
        "connected_ms": bot.connected_ms,
        "invite_url": invite::url(&bot.application_id),
        "status": status.name(),
    });
    if let Status::Online { world } = status {
        let name = s
            .worlds
            .read()
            .await
            .get(&world)
            .and_then(|r| r.body["name"].as_str().map(str::to_string))
            .unwrap_or(world);
        wire["world"] = json!(name);
    }
    wire
}

/// A Discord refusal as the console reads it.
fn refused(npc_id: u64, e: DiscordError) -> Response {
    tracing::warn!(npc = npc_id, error = %e, "discord: connect refused");
    let detail = e.to_string();
    match e {
        DiscordError::TokenRejected => {
            err(StatusCode::BAD_REQUEST, "discord_token_rejected", &detail)
        }
        DiscordError::NotABot => err(StatusCode::BAD_REQUEST, "discord_not_a_bot", &detail),
        DiscordError::RateLimited { retry_after_s } | DiscordError::Paused { retry_after_s } => {
            let mut res = (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "error": "discord_rate_limited",
                    "detail": detail,
                    "retry_after": retry_after_s,
                })),
            )
                .into_response();
            let whole = retry_after_s.ceil().max(1.0) as u64;
            res.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(whole));
            res
        }
        DiscordError::Status { .. } => err(StatusCode::BAD_GATEWAY, "discord_failed", &detail),
        DiscordError::Transport(_) => err(StatusCode::BAD_GATEWAY, "discord_unreachable", &detail),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::env::temp_dir;
    use std::fs::{create_dir_all, remove_dir_all};
    use std::path::PathBuf;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use axum::Router;
    use tower::ServiceExt;
    use web::auth::session::Identity;

    use super::*;
    use crate::accounts::{self, Accounts};
    use crate::api::router;
    use crate::collections::Libraries;
    use crate::discord::client::Client;
    use crate::discord::fake::{self, Mode, Seen};
    use crate::discord::state::{BotState, State as BotStateKind};
    use crate::images::Images;
    use crate::mind::Mind;
    use crate::npcs::Npcs;
    use crate::projection::Source;
    use crate::registry::Registry;

    const OWNER: &str = "google-1";
    const STRANGER: &str = "google-2";
    const TOKEN: &str = "MTEx.bot.token";

    fn tmp() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = temp_dir().join(format!(
            "npcd-discord-routes-{}-{}",
            process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = remove_dir_all(&p);
        create_dir_all(&p).unwrap();
        p
    }

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

    /// npcd with no engine, talking to `discord`, and [`OWNER`]'s character
    /// Zen in it — with a portrait when `portrait` is given. The state and
    /// Zen's wire id.
    fn state(discord: &str, portrait: Option<&[u8]>) -> (Arc<Authored>, String) {
        let base = tmp();
        let npcs = Npcs::load(&base).unwrap();
        let accounts = Accounts::open(npcs.substrate(), &base.join("accounts")).unwrap();
        let (images, _) = Images::open(npcs.substrate(), &base.join("images")).unwrap();
        let mut authored = Authored::new(
            Registry::load("world", base.join("worlds")).unwrap(),
            Registry::load("personality", base.join("personalities")).unwrap(),
            accounts,
            npcs,
            serde_yaml::from_str(&format!(
                "people:\n  - sub: {OWNER}\n    role: user\n  - sub: {STRANGER}\n    role: user\n"
            ))
            .unwrap(),
            Libraries::load(&Source::resolve(None).unwrap()),
            Mind::new(None),
            images,
            BTreeMap::new(),
        );
        authored.discord = Client::at(discord);
        let owner = accounts::user_id_of(&identity(OWNER));
        let zen = authored
            .npcs
            .create(
                &identity(OWNER),
                &owner,
                &json!({ "name": "Zen", "world_id": "life", "personality_id": "zen" }),
                1,
            )
            .unwrap();
        let wire = zen["npc_id"].as_str().unwrap().to_string();
        if let Some(png) = portrait {
            let iid = authored.images.put(png).unwrap();
            authored
                .npcs
                .set_portrait(
                    npcs::npc_id_of_wire(&wire).unwrap(),
                    &owner,
                    iid,
                    "uploaded",
                    2,
                )
                .unwrap();
        }
        (Arc::new(authored), wire)
    }

    fn request(method: &str, path: &str, sub: &str, body: Option<Value>) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .header("x-tokera-user", sub)
            .header("x-tokera-provider", "google")
            .header("x-tokera-email", "wren@example.com")
            .header("x-tokera-name", "Wren S")
            .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
            .unwrap()
    }

    /// The status, the body as JSON, and the body as it was sent.
    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value, String) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        (
            status,
            serde_json::from_str(&text).unwrap_or(Value::Null),
            text,
        )
    }

    fn png() -> Vec<u8> {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[7u8; 32]);
        png
    }

    /// **The owner connects a bot, sees it, and disconnects it**; the bot is
    /// named and pictured as the character, and its token is sealed into the
    /// record and in no answer.
    #[tokio::test]
    async fn the_owner_connects_sees_and_disconnects_a_bot() {
        let discord = fake::start(Mode::Ok).await;
        let png = png();
        let (st, zen) = state(&discord.url, Some(&png));
        let path = format!("/v1/npc/{zen}/discord");

        let (s, v, _) = call(router(st.clone()), request("GET", &path, OWNER, None)).await;
        assert_eq!((s, v), (StatusCode::OK, json!({ "connected": false })));

        let (s, v, text) = call(
            router(st.clone()),
            request(
                "PUT",
                &path,
                OWNER,
                Some(json!({ "token": format!("  {TOKEN}\n") })),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let connected_ms = v["connected_ms"].as_u64().expect("a time");
        let expected = json!({
            "connected": true,
            "application_id": "111",
            "user_id": "222",
            "username": "Zen",
            "connected_ms": connected_ms,
            "invite_url": "https://discord.com/oauth2/authorize?client_id=111&scope=bot&permissions=344134372416",
            "status": "offline",
        });
        assert_eq!(v, expected);
        assert!(!text.contains(TOKEN), "{text}");

        // Checked, then named and pictured — the token trimmed of what the
        // paste carried.
        let as_bot = |method: &str, path: &str, body: Value| Seen {
            method: method.into(),
            path: path.into(),
            authorization: format!("Bot {TOKEN}"),
            user_agent: discord.seen()[0].user_agent.clone(),
            body,
        };
        assert_eq!(
            discord.seen(),
            [
                as_bot("GET", "/users/@me", Value::Null),
                as_bot("GET", "/applications/@me", Value::Null),
                as_bot(
                    "PATCH",
                    "/users/@me",
                    json!({ "username": "Zen", "avatar": profile::avatar(&png, "image/png") })
                ),
            ]
        );

        // The record holds the token sealed for this character, and only so.
        let npc_id = npcs::npc_id_of_wire(&zen).unwrap();
        let bot = store::get(&st.npcs.substrate(), npc_id).unwrap();
        assert_ne!(bot.token, TOKEN);
        assert_eq!(
            st.discord_tokens.open(npc_id, &bot.token),
            Ok(TOKEN.to_string())
        );

        let (s, v, text) = call(router(st.clone()), request("GET", &path, OWNER, None)).await;
        assert_eq!((s, v), (StatusCode::OK, expected));
        assert!(
            !text.contains(TOKEN) && !text.contains(&bot.token),
            "{text}"
        );

        let (s, v, _) = call(router(st.clone()), request("DELETE", &path, OWNER, None)).await;
        assert_eq!((s, v), (StatusCode::OK, json!({ "connected": false })));
        let (_, v, _) = call(router(st.clone()), request("GET", &path, OWNER, None)).await;
        assert_eq!(v, json!({ "connected": false }));
        assert_eq!(store::get(&st.npcs.substrate(), npc_id), None);
    }

    /// **The read says what the bot's state record says** while it is in
    /// force: online with the world it is in, or refused.
    #[tokio::test]
    async fn the_read_carries_the_bots_status() {
        let discord = fake::start(Mode::Ok).await;
        let (st, zen) = state(&discord.url, None);
        let npc_id = npcs::npc_id_of_wire(&zen).unwrap();
        let path = format!("/v1/npc/{zen}/discord");
        let bot = Bot {
            application_id: "111".into(),
            user_id: "222".into(),
            username: "Zen".into(),
            token: "c2VhbGVk".into(),
            connected_ms: 10,
        };
        store::put(&st.npcs.substrate(), npc_id, &bot).unwrap();
        let shared = st.npcs.substrate();
        for (kind, status, world) in [
            (BotStateKind::Online, "online", Some("life")),
            (BotStateKind::Refused, "refused", None),
        ] {
            state::put(
                &shared,
                npc_id,
                &BotState {
                    state: kind,
                    since_ms: 20,
                    world: "life".into(),
                    connected_ms: 10,
                },
            )
            .unwrap();
            let (_, v, _) = call(router(st.clone()), request("GET", &path, OWNER, None)).await;
            assert_eq!(v["status"], status);
            assert_eq!(v["world"].as_str(), world);
        }
        state::remove(&shared, npc_id).unwrap();
        let (_, v, _) = call(router(st.clone()), request("GET", &path, OWNER, None)).await;
        assert_eq!(v["status"], "offline");
    }

    /// **A bot already named as its character, of a character with no
    /// portrait, is not renamed**: Discord allows a bot few renames an hour.
    #[tokio::test]
    async fn nothing_that_already_matches_is_sent() {
        let discord = fake::start(Mode::Ok).await;
        let (st, zen) = state(&discord.url, None);
        let npc_id = npcs::npc_id_of_wire(&zen).unwrap();
        let owner = accounts::user_id_of(&identity(OWNER));
        st.npcs
            .patch(npc_id, &owner, &json!({ "name": fake::OLD_NAME }), 3)
            .unwrap();
        let (s, v, _) = call(
            router(st.clone()),
            request(
                "PUT",
                &format!("/v1/npc/{zen}/discord"),
                OWNER,
                Some(json!({ "token": TOKEN })),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["username"], fake::OLD_NAME);
        let paths: Vec<(String, String)> = discord
            .seen()
            .into_iter()
            .map(|r| (r.method, r.path))
            .collect();
        assert_eq!(
            paths,
            [
                ("GET".to_string(), "/users/@me".to_string()),
                ("GET".to_string(), "/applications/@me".to_string()),
            ]
        );
    }

    /// **Somebody else's character is no character at all**: 404 on every
    /// method, and not a request reaches Discord.
    #[tokio::test]
    async fn a_stranger_gets_the_404_of_a_missing_character() {
        let discord = fake::start(Mode::Ok).await;
        let (st, zen) = state(&discord.url, None);
        let path = format!("/v1/npc/{zen}/discord");
        for (method, body) in [
            ("GET", None),
            ("PUT", Some(json!({ "token": TOKEN }))),
            ("DELETE", None),
        ] {
            let (s, v, _) = call(router(st.clone()), request(method, &path, STRANGER, body)).await;
            assert_eq!(
                (s, v["error"].as_str()),
                (StatusCode::NOT_FOUND, Some("npc_not_found")),
                "{method}"
            );
        }
        let (s, v, _) = call(
            router(st.clone()),
            request("GET", "/v1/npc/zzzzzz/discord", OWNER, None),
        )
        .await;
        assert_eq!(
            (s, v["error"].as_str()),
            (StatusCode::NOT_FOUND, Some("npc_not_found"))
        );
        assert_eq!(discord.seen(), []);
    }

    /// **A token Discord refuses is said so, and nothing is kept**; a rate
    /// limit says how long to wait.
    #[tokio::test]
    async fn a_refused_token_is_said_so_and_nothing_is_kept() {
        for (mode, status, code) in [
            (
                Mode::Unauthorized,
                StatusCode::BAD_REQUEST,
                "discord_token_rejected",
            ),
            (Mode::NotABot, StatusCode::BAD_REQUEST, "discord_not_a_bot"),
            (
                Mode::RateLimited,
                StatusCode::TOO_MANY_REQUESTS,
                "discord_rate_limited",
            ),
            (Mode::Failing, StatusCode::BAD_GATEWAY, "discord_failed"),
        ] {
            let discord = fake::start(mode).await;
            let (st, zen) = state(&discord.url, None);
            let res = router(st.clone())
                .oneshot(request(
                    "PUT",
                    &format!("/v1/npc/{zen}/discord"),
                    OWNER,
                    Some(json!({ "token": TOKEN })),
                ))
                .await
                .unwrap();
            assert_eq!(res.status(), status, "{mode:?}");
            let retry = res.headers().get(header::RETRY_AFTER).cloned();
            let v: Value =
                serde_json::from_slice(&to_bytes(res.into_body(), 1 << 20).await.unwrap()).unwrap();
            assert_eq!(v["error"], code, "{mode:?}");
            if mode == Mode::RateLimited {
                assert_eq!(v["retry_after"], 2.5);
                assert_eq!(retry, Some(HeaderValue::from(3u64)));
            }
            assert_eq!(
                store::get(&st.npcs.substrate(), npcs::npc_id_of_wire(&zen).unwrap()),
                None,
                "{mode:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_blank_token_is_refused_before_discord_is_asked() {
        let discord = fake::start(Mode::Ok).await;
        let (st, zen) = state(&discord.url, None);
        let (s, v, _) = call(
            router(st),
            request(
                "PUT",
                &format!("/v1/npc/{zen}/discord"),
                OWNER,
                Some(json!({ "token": "   " })),
            ),
        )
        .await;
        assert_eq!(
            (s, v["error"].as_str()),
            (StatusCode::BAD_REQUEST, Some("discord_token_missing"))
        );
        assert_eq!(discord.seen(), []);
    }
}
