//! A world mirrored into its own Discord server (`docs/npcd_discord.md`
//! §3.1, §4.4, §5), against the fake gateway and REST API: `#the-channel`
//! made and its webhook, what the world says posted by the speaker's own bot
//! or the webhook under the speaker's name, a Creator's post on Discord heard
//! on the channel and not posted back, everyone else's dropped, and the
//! records the world keeps of it.

use serde_json::{json, Value};

use super::*;
use crate::accounts::public_id;
use crate::discord::fake::{FIRST_CHANNEL, HOOK_OFFSET, HOOK_TOKEN};
use crate::discord::trace::note::Direction;
use crate::discord::world::channels::CHANNEL_NAME;
use crate::discord::world::record::{hook_name, Guild, Kind};
use crate::engine::speaking_as;
use crate::npcs::npc_id_wire;
use crate::sim::phone::CHANNEL as THE_CHANNEL;

/// The world's server, a channel it already has, and the channel the
/// listener makes for the world's open channel.
const GUILD: u64 = 600;
const GENERAL: u64 = 610;
const MADE: u64 = FIRST_CHANNEL;
/// The Creator's Discord user, and somebody linked to nothing.
const ADA: u64 = 99;
const KIT: u64 = 88;

/// A message in the made channel from `author`.
fn posted_in_world(id: u64, author: u64, content: &str) -> Value {
    json!({
        "id": id.to_string(),
        "channel_id": MADE.to_string(),
        "guild_id": GUILD.to_string(),
        "author": { "id": author.to_string(), "username": "someone" },
        "content": content,
    })
}

/// A Discord message id from a moment after now.
fn later() -> u64 {
    snowflake_at(now_ms() + 1_000)
}

impl Daemon {
    /// The world's server set, with Zen its listener and Zen and Wick on the
    /// world's channel.
    fn serve_world(&self) {
        self.discord.channels(&[(GENERAL, 0, "general")]);
        self.hosted.with_sim(|s| {
            s.enrol("Zen");
            s.enrol("Wick");
        });
        self.hosted.with_mirror(|m| {
            m.set_guild(Some(Guild {
                guild_id: GUILD,
                name: "Life".into(),
                listener: self.zen,
                backstage: true,
            }))
        });
    }

    /// The world's channel's thread.
    fn the_channel(&self) -> String {
        self.hosted
            .sim(|s| {
                s.threads
                    .iter()
                    .find(|t| t.permanent && t.name == THE_CHANNEL)
                    .map(|t| t.id.clone())
            })
            .expect("the world's channel")
    }

    /// `who` says `text` on the world's channel.
    fn on_channel(&self, who: &str, text: &str) {
        self.hosted
            .with_sim(|s| s.threads.send(who, THE_CHANNEL, text).unwrap());
    }

    /// What the world's channel carries, as `(from, text, from_discord)`.
    fn channel_said(&self) -> Vec<(String, String, bool)> {
        let thread = self.the_channel();
        self.hosted.sim(|s| {
            s.threads.get(&thread).map_or_else(Vec::new, |t| {
                t.messages
                    .iter()
                    .map(|m| (m.from.clone(), m.intent.clone(), m.origin.is_some()))
                    .collect()
            })
        })
    }

    /// The webhook posts the fake was sent, as `(username, content)`.
    fn hooked(&self) -> Vec<(String, String)> {
        self.discord
            .seen()
            .into_iter()
            .filter(|s| s.path.starts_with("/webhooks/"))
            .map(|s| {
                (
                    s.body["username"].as_str().unwrap_or_default().into(),
                    s.body["content"].as_str().unwrap_or_default().into(),
                )
            })
            .collect()
    }

    /// Online, with `#the-channel` made and its webhook.
    async fn mirroring(&self) {
        self.serve_world();
        self.online().await;
        self.until("the channel mirrored", |d| {
            d.hosted.mirror(|m| m.hook(MADE).is_some())
        })
        .await;
    }
}

/// **The listener makes `#the-channel` when the server has none, and a
/// webhook on it**, records the channel with its cursor at the thread's end
/// — what the world said before is not posted — and keeps the webhook's
/// token only sealed among the world's records.
#[tokio::test]
async fn the_listener_makes_the_channel_and_its_webhook() {
    let d = daemon(vec![Script::Serve]).await;
    d.serve_world();
    d.on_channel("Wick", "said before the server was set");
    d.online().await;
    d.until("the channel mirrored", |d| {
        d.hosted.mirror(|m| m.hook(MADE).is_some())
    })
    .await;

    let token = |path: &str, body: Value| as_bot_of(path, body);
    let calls: Vec<Seen> = d
        .discord
        .seen()
        .into_iter()
        .filter(|s| !s.path.contains("/messages"))
        .collect();
    assert_eq!(
        calls,
        [
            token("GET /guilds/600/channels", Value::Null),
            token(
                "POST /guilds/600/channels",
                json!({ "name": CHANNEL_NAME, "type": 0 })
            ),
            token(
                &format!("POST /channels/{MADE}/webhooks"),
                json!({ "name": "Life" })
            ),
        ]
    );
    let thread = d.the_channel();
    let mirrored = d.hosted.mirror(|m| m.thread(&thread)).expect("recorded");
    assert_eq!(
        (
            mirrored.channel_id,
            mirrored.kind,
            mirrored.mirrored_through
        ),
        (MADE, Kind::Channel, 1)
    );
    d.settle().await;
    assert_eq!(d.hooked(), [], "what was said before the server was posted");

    d.rt.beat(&d.hosted);
    let records = world_store::read(&d.a.npcs.substrate(), WORLD);
    assert_eq!(
        records.get("discord/guild"),
        Some(&json!({ "guild_id": GUILD, "name": "Life", "listener": d.zen, "backstage": true }))
    );
    assert_eq!(
        records.get(&format!("discord/thread/{thread}")),
        Some(&json!({
            "channel_id": MADE,
            "kind": "channel",
            "mirrored_through": 1,
            "last_seen_discord": mirrored.last_seen_discord,
        }))
    );
    let hook = &records[&format!("discord/webhook/{MADE}")];
    assert_eq!(hook["webhook_id"], json!(MADE + HOOK_OFFSET));
    let sealed = hook["token"].as_str().unwrap();
    assert_ne!(sealed, HOOK_TOKEN, "the webhook's token was kept open");
    assert_eq!(
        d.a.discord_tokens.open_for(&hook_name(MADE), sealed),
        Ok(HOOK_TOKEN.to_string())
    );
}

/// **A channel the server already has is used, not made again.**
#[tokio::test]
async fn a_channel_the_server_has_is_found_not_made() {
    let d = daemon(vec![Script::Serve]).await;
    d.serve_world();
    d.discord
        .channels(&[(GENERAL, 0, "general"), (650, 0, CHANNEL_NAME)]);
    d.online().await;
    d.until("the channel mirrored", |d| {
        d.hosted.mirror(|m| m.hook(650).is_some())
    })
    .await;
    assert!(
        !d.discord
            .seen()
            .iter()
            .any(|s| s.method == "POST" && s.path == "/guilds/600/channels"),
        "a channel was made beside the one the server has"
    );
}

/// **What the world says is posted by its speaker**: a character without a
/// bot in the server, and Zen before its bot is in it, through the webhook
/// under their own names; Zen, once its bot is in the server, as its bot —
/// each once, in order.
#[tokio::test]
async fn the_world_is_posted_by_its_speakers() {
    let d = daemon(vec![Script::Serve]).await;
    d.mirroring().await;

    d.on_channel("Wick", "the hatchery is warm");
    d.on_channel("Zen", "good");
    d.until("both posted", |d| d.hooked().len() == 2).await;
    assert_eq!(
        d.hooked(),
        [
            ("Wick".to_string(), "the hatchery is warm".to_string()),
            ("Zen".to_string(), "good".to_string()),
        ]
    );

    // Zen's bot learns of the server, which records it there.
    d.gateway.event(
        "GUILD_CREATE",
        json!({ "id": GUILD.to_string(), "name": "Life", "channels": [
            { "id": MADE.to_string(), "type": 0, "name": CHANNEL_NAME },
        ]}),
    );
    d.until("the server recorded", |d| {
        d.hosted.servers(|s| s.server(d.zen, GUILD)).is_some()
    })
    .await;
    assert_eq!(
        d.hosted.servers(|s| s.channels_in(d.zen, GUILD)),
        [],
        "the world's server became a thread of Zen's own"
    );
    d.on_channel("Zen", "I am watching");
    d.until("posted as the bot", |d| {
        d.posts()
            .iter()
            .any(|(path, _)| path == &format!("/channels/{MADE}/messages"))
    })
    .await;
    d.settle().await;
    assert_eq!(
        d.posts(),
        [(
            format!("/channels/{MADE}/messages"),
            "I am watching".to_string()
        )]
    );
    assert_eq!(d.hooked().len(), 2, "something was posted twice");
    let thread = d.the_channel();
    assert_eq!(
        d.hosted
            .mirror(|m| m.thread(&thread))
            .map(|m| m.mirrored_through),
        Some(3)
    );
}

/// **A Creator's post on Discord is heard on the world's channel as them,
/// and not posted back**; an unlinked author's, a linked author's without
/// the role, a webhook's and another bot's are dropped, and none of them
/// is read twice.
#[tokio::test]
async fn a_creators_post_is_heard_and_the_rest_are_dropped() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(CREATOR, ADA).await;
    d.link(OWNER, WREN).await;
    d.mirroring().await;

    let first = later();
    d.gateway
        .dispatch(posted_in_world(first, ADA, "the moon is quiet tonight"));
    d.gateway
        .dispatch(posted_in_world(first + 1, KIT, "let me in"));
    d.gateway
        .dispatch(posted_in_world(first + 2, WREN, "and me"));
    let mut hook = posted_in_world(first + 3, MADE + HOOK_OFFSET, "an echo");
    hook["webhook_id"] = json!((MADE + HOOK_OFFSET).to_string());
    d.gateway.dispatch(hook);
    let mut other = posted_in_world(first + 4, 5, "beep");
    other["author"]["bot"] = json!(true);
    d.gateway.dispatch(other);
    d.gateway
        .dispatch(posted_in_world(first, ADA, "the moon is quiet tonight"));

    let ada = {
        let accounts = d.a.accounts.read().await;
        let id = accounts.identity(&key_of(&identity(CREATOR))).unwrap();
        speaking_as(&id, &public_id(&key_of(&identity(CREATOR))), &d.a.roles)
    };
    d.until("the Creator heard", |d| {
        d.channel_said().iter().any(|(from, _, _)| *from == ada)
    })
    .await;
    d.settle().await;
    assert_eq!(
        d.channel_said(),
        [(ada, "the moon is quiet tonight".to_string(), true)]
    );
    assert_eq!(d.hooked(), [], "a post from Discord was posted back");
    let thread = d.the_channel();
    assert_eq!(
        d.hosted
            .mirror(|m| m.thread(&thread))
            .map(|m| m.last_seen_discord),
        Some(first + 2),
        "the cursor stopped short of what was judged"
    );
    let outcomes: Vec<String> =
        d.a.bots
            .trace()
            .notes(d.zen)
            .into_iter()
            .filter(|n| n.kind == "mirror" && n.direction == Direction::In)
            .map(|n| n.outcome)
            .collect();
    for outcome in [
        "applied",
        "unlinked_dropped",
        "not_creator_dropped",
        "bot_or_webhook_dropped",
        "duplicate",
    ] {
        assert!(
            outcomes.iter().any(|o| o == outcome),
            "{outcome}: {outcomes:?}"
        );
    }
}

/// **Every other bot drops the world's server**: a message there on a shard
/// that is not the listener's is heard by nobody, and stands no thread of
/// the character's own.
#[tokio::test]
async fn a_bot_that_is_not_the_listener_drops_the_worlds_server() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(CREATOR, ADA).await;
    d.serve_world();
    d.hosted.with_mirror(|m| {
        m.set_guild(Some(Guild {
            guild_id: GUILD,
            name: "Life".into(),
            listener: d.zen + 1,
            backstage: true,
        }))
    });
    d.online().await;
    d.gateway.dispatch(posted_in_world(later(), ADA, "anyone?"));
    d.settle().await;
    assert_eq!(
        d.channel_said(),
        [],
        "a bot that is not the listener heard the world's server"
    );
    assert!(d.hosted.mirror(|m| m.threads().is_empty()));
    assert_eq!(d.hosted.servers(|s| s.channels_of(d.zen)), []);
}

/// `method` on `/v1/world/<WORLD>/discord` as `sub`, through the engine's
/// routes and their role check; the status and the body.
async fn world_route(d: &Daemon, method: &str, sub: &str, body: Option<Value>) -> (u16, Value) {
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    let app = crate::engine::api(d.a.roles.clone())
        .into_routes(d.a.clone())
        .router;
    let req = Request::builder()
        .method(method)
        .uri(format!("/v1/world/{WORLD}/discord"))
        .header("content-type", "application/json")
        .header("x-tokera-user", sub)
        .header("x-tokera-provider", "google")
        .header("x-tokera-email", format!("{sub}@example.com"))
        .header("x-tokera-name", "Ada")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    let status = res.status().as_u16();
    let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// **A Creator sets the world's server, reads it and takes it away**, once
/// the listener is a character with a body in the world and a bot that is in
/// the server — whose name Discord's answer gives. Each check that fails is
/// said by name, and a person without the role is refused before any.
#[tokio::test]
async fn a_creator_sets_reads_and_takes_away_the_worlds_server() {
    let d = daemon(vec![Script::Serve]).await;
    let zen = npc_id_wire(d.zen);
    let set = |listener: &str, guild: &str| {
        Some(json!({ "guild_id": guild, "listener": listener, "backstage": false }))
    };

    assert_eq!(
        world_route(&d, "GET", CREATOR, None).await,
        (200, json!({ "guild": null }))
    );
    let (status, _) = world_route(&d, "PUT", OWNER, set(&zen, "600")).await;
    assert_eq!(status, 403, "a user set the world's server");

    let nobody = npc_id_wire(d.zen ^ 1);
    let (status, v) = world_route(&d, "PUT", CREATOR, set(&nobody, "600")).await;
    assert_eq!(
        (status, v["error"].clone()),
        (400, json!("listener_not_in_world"))
    );
    let (status, v) = world_route(&d, "PUT", CREATOR, set(&zen, "six hundred")).await;
    assert_eq!(
        (status, v["error"].clone()),
        (400, json!("discord_guild_id_invalid"))
    );
    d.discord
        .reply(403, json!({ "message": "Missing Access", "code": 50001 }));
    let (status, v) = world_route(&d, "PUT", CREATOR, set(&zen, "600")).await;
    assert_eq!(
        (status, v["error"].clone()),
        (400, json!("listener_not_in_server"))
    );
    assert!(
        d.hosted.mirror(|m| m.guild().is_none()),
        "a refused server was set"
    );

    let guild = json!({ "guild": {
        "guild_id": "600", "name": crate::discord::fake::GUILD_NAME,
        "listener": zen, "backstage": false,
    }});
    assert_eq!(
        world_route(&d, "PUT", CREATOR, set(&zen, " 600 ")).await,
        (200, guild.clone())
    );
    assert_eq!(world_route(&d, "GET", CREATOR, None).await, (200, guild));
    assert_eq!(
        d.hosted.mirror(|m| m.guild().cloned()),
        Some(Guild {
            guild_id: GUILD,
            name: crate::discord::fake::GUILD_NAME.into(),
            listener: d.zen,
            backstage: false,
        })
    );
    assert_eq!(
        world_route(&d, "DELETE", CREATOR, None).await,
        (200, json!({ "guild": null }))
    );
    assert!(d.hosted.mirror(|m| m.guild().is_none()));
}

/// A request as Zen's bot: `"<METHOD> <path>"` and its body.
fn as_bot_of(call: &str, body: Value) -> Seen {
    let (method, path) = call.split_once(' ').unwrap();
    Seen {
        method: method.into(),
        path: path.into(),
        authorization: format!("Bot {TOKEN}"),
        user_agent: concat!(
            "DiscordBot (https://tokera.com, ",
            env!("CARGO_PKG_VERSION"),
            ")"
        )
        .into(),
        body,
    }
}
