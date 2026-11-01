//! A bot in its servers' channels (`docs/npcd_discord.md` §3.4), against
//! the fake gateway and REST API: a server joined and told, its channels'
//! threads opened, a message in each way, renames, a removal, what `READY`
//! says of the servers left and joined while offline, and a `4014`.

use axum::body::to_bytes;
use axum::extract::{Path as Wire, State as Shared};
use axum::http::StatusCode;
use serde_json::{json, Value};

use super::*;
use crate::authority::ops::{OpReply, WorldOp};
use crate::discord::fake::FIRST_POSTED;
use crate::discord::gateway::fake::SESSION;
use crate::discord::server::record::{Channel, Server};
use crate::discord::trace::note::Direction;
use crate::discord::trace::route::get_trace;
use crate::engine::event::EventKind;
use crate::npcs::npc_id_wire;
use crate::sim::phone::Thread;

/// The server the bot is added to, and its text channels.
const GUILD: u64 = 600;
const GENERAL: u64 = 610;
const LORE: u64 = 611;
/// Somebody on Discord who has linked no account.
const KIT: u64 = 88;

/// A `GUILD_CREATE` for server `id` called `name`, with `channels` as
/// `(id, type, name)`.
fn guild(id: u64, name: &str, channels: &[(u64, u8, &str)]) -> Value {
    json!({
        "id": id.to_string(),
        "name": name,
        "channels": channels
            .iter()
            .map(|(id, kind, name)| json!({ "id": id.to_string(), "type": kind, "name": name }))
            .collect::<Vec<_>>(),
    })
}

/// The server Zen's bot is added to: two text channels and a voice one.
fn zenworld() -> Value {
    guild(
        GUILD,
        "zenworld",
        &[
            (GENERAL, 0, "general"),
            (620, 2, "Lounge"),
            (LORE, 0, "lore"),
        ],
    )
}

/// A Discord message id from a moment after now, as anything written once
/// the bot has learned of a channel has.
fn later() -> u64 {
    snowflake_at(now_ms() + 1_000)
}

/// A message in `#general` from `author`, called `username`.
fn said(id: u64, author: u64, username: &str, content: &str) -> Value {
    json!({
        "id": id.to_string(),
        "channel_id": GENERAL.to_string(),
        "guild_id": GUILD.to_string(),
        "author": { "id": author.to_string(), "username": username },
        "content": content,
    })
}

impl Daemon {
    /// Zen in the scheduler, so what it is told can be read back.
    fn listen(&self) {
        self.rt.scheduler.wake(self.zen, 0, 0);
    }

    /// Everything Zen has been told since last asked.
    fn told(&self) -> Vec<EventKind> {
        self.rt
            .scheduler
            .begin_tick(self.zen)
            .map(|t| t.events.into_iter().map(|e| e.kind).collect())
            .unwrap_or_default()
    }

    /// The descriptions Zen has been told since last asked.
    fn notices(&self) -> Vec<String> {
        self.told()
            .into_iter()
            .filter_map(|k| match k {
                EventKind::Description { text } => Some(text),
                _ => None,
            })
            .collect()
    }

    fn record(&self, channel: u64) -> Option<Channel> {
        self.hosted.servers(|s| s.channel(self.zen, channel))
    }

    /// The thread mirroring `channel`.
    fn channel_thread(&self, channel: u64) -> Option<Thread> {
        let record = self.record(channel)?;
        self.hosted.sim(|s| s.threads.get(&record.thread).cloned())
    }

    /// What is said on the thread mirroring `channel`, as `(from, text)`.
    fn heard(&self, channel: u64) -> Vec<(String, String)> {
        self.channel_thread(channel)
            .map(|t| {
                t.messages
                    .iter()
                    .map(|m| (m.from.clone(), m.intent.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Online, then added to zenworld, and told so.
    async fn joined(&self) {
        self.listen();
        self.online().await;
        self.gateway.event("GUILD_CREATE", zenworld());
        self.until("joined", |d| d.record(LORE).is_some()).await;
    }
}

/// **A server the bot is added to is told to the character, and each text
/// channel becomes a standing group thread** with the character on it —
/// the voice channel not — recorded as world records with the server's
/// name.
#[tokio::test]
async fn a_join_is_told_and_opens_the_threads() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    assert_eq!(
        d.notices(),
        [
            "Someone added you to the Discord server zenworld. You can talk in its channels: \
          zenworld #general, #lore"
        ]
    );
    for (channel, name) in [(GENERAL, "zenworld #general"), (LORE, "zenworld #lore")] {
        let t = d.channel_thread(channel).expect("a thread");
        assert_eq!(
            (
                t.name.as_str(),
                t.members.as_slice(),
                t.open,
                t.named,
                t.permanent
            ),
            (name, ["Zen".to_string()].as_slice(), true, true, true)
        );
    }
    assert_eq!(d.hosted.servers(|s| s.channel(d.zen, 620)), None);
    let mut names = d.hosted.sim(|s| s.threads.names_for("Zen"));
    names.sort();
    assert_eq!(names, ["zenworld #general", "zenworld #lore"]);

    d.rt.beat(&d.hosted);
    let records = world_store::read(&d.a.npcs.substrate(), WORLD);
    let general = d.record(GENERAL).unwrap();
    assert_eq!(
        records.get(&format!("discord/channel/{}/{GENERAL}", d.zen)),
        Some(&json!({
            "guild_id": GUILD,
            "thread": general.thread,
            "mirrored_through": 0,
            "last_seen_discord": general.last_seen_discord,
        }))
    );
    assert_eq!(
        records.get(&format!("discord/server/{}/{GUILD}", d.zen)),
        Some(&json!({ "name": "zenworld" }))
    );
    assert_eq!(d.discord.seen(), [], "a join posted something");
}

/// **A message from anyone the server lets write is heard on the channel's
/// thread** — an unlinked author as their display name and username, a
/// linked one as the name their account speaks by — each joined to the
/// thread on their first message, mentions written out, and the character
/// handed it as a message on the group, which wakes it.
#[tokio::test]
async fn a_guild_message_is_heard_from_who_wrote_it_and_wakes_the_character() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.joined().await;
    d.told();

    // Written after the bot learned of the channel, so above its cursor.
    let first = later();
    let mut kit = said(first, KIT, "kit", "<@222> do you know <@!77>?");
    kit["author"]["global_name"] = json!("Kit");
    kit["mentions"] = json!([
        { "id": BOT_USER.to_string(), "username": "Zen", "bot": true },
        { "id": WREN.to_string(), "username": "wren", "member": { "nick": "Wrennie" } },
    ]);
    d.gateway.dispatch(kit);
    d.gateway
        .dispatch(said(first + 1, WREN, "wren", "I'm here too"));
    d.until("heard", |d| d.heard(GENERAL).len() == 2).await;

    assert_eq!(
        d.heard(GENERAL),
        [
            (
                "Kit (@kit)".to_string(),
                "@Zen do you know @Wrennie?".to_string()
            ),
            (WREN_S.to_string(), "I'm here too".to_string()),
        ]
    );
    let t = d.channel_thread(GENERAL).unwrap();
    assert_eq!(t.members, ["Zen", "Kit (@kit)", WREN_S]);
    assert_eq!(
        t.messages.iter().map(|m| m.origin).collect::<Vec<_>>(),
        [
            Some(Origin::Discord {
                message: first,
                channel: GENERAL
            }),
            Some(Origin::Discord {
                message: first + 1,
                channel: GENERAL
            }),
        ]
    );
    assert_eq!(d.record(GENERAL).unwrap().last_seen_discord, first + 1);

    // The world's moment hands them to the character on the group.
    d.rt.beat(&d.hosted);
    let handed: Vec<(String, String, String)> = d
        .told()
        .into_iter()
        .filter_map(|k| match k {
            EventKind::Message { thread, from, text } => Some((thread, from, text)),
            _ => None,
        })
        .collect();
    assert_eq!(
        handed,
        [
            (
                "zenworld #general".to_string(),
                "Kit (@kit)".to_string(),
                "@Zen do you know @Wrennie?".to_string()
            ),
            (
                "zenworld #general".to_string(),
                WREN_S.to_string(),
                "I'm here too".to_string()
            ),
        ]
    );
    d.settle().await;
    assert_eq!(
        d.discord.seen(),
        [],
        "a message from Discord was posted back"
    );
}

/// **Anything a bot or a webhook wrote is dropped** — this bot's own echo
/// among them — and a message already heard is not heard again.
#[tokio::test]
async fn bot_and_webhook_messages_are_dropped_and_none_is_heard_twice() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    let id = later();
    d.gateway
        .dispatch(said(id, BOT_USER, "Zen", "my own post, echoed"));
    let mut another_bot = said(id + 1, 99, "helper", "beep");
    another_bot["author"]["bot"] = json!(true);
    d.gateway.dispatch(another_bot);
    let mut hook = said(id + 2, KIT, "kit", "via a hook");
    hook["webhook_id"] = json!("5");
    d.gateway.dispatch(hook);
    d.gateway.dispatch(said(id + 3, KIT, "kit", "the real one"));
    d.gateway.dispatch(said(id + 3, KIT, "kit", "the real one"));
    // And one from before the bot learned of the channel: the channel's.
    d.gateway.dispatch(said(901, KIT, "kit", "ancient history"));
    d.until("heard", |d| !d.heard(GENERAL).is_empty()).await;
    d.settle().await;
    assert_eq!(
        d.heard(GENERAL),
        [("kit (@kit)".to_string(), "the real one".to_string())]
    );
    assert_eq!(d.discord.seen(), []);
}

/// **The character's `message` act on `zenworld #general` is posted** to the
/// channel as the bot, with exactly the request Discord documents; what a
/// character it invited says there is posted as a quoted line; and the
/// cursor goes with the world's records.
#[tokio::test]
async fn the_characters_message_on_a_channel_is_posted() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    let call = json!({
        "tool": "message",
        "via": "channel",
        "to": "zenworld #general",
        "intent": "Hello, zenworld.",
    })
    .to_string();
    let reply =
        d.rt.make(
            WORLD,
            WorldOp::Act {
                npc: d.zen,
                call,
                decided: None,
                pinned_at: None,
            },
            None,
        )
        .await
        .unwrap();
    assert!(
        matches!(reply, OpReply::Performed { landed: true, .. }),
        "{reply:?}"
    );
    let thread = d.record(GENERAL).unwrap().thread;
    d.hosted.with_sim(|s| {
        s.threads.join(&thread, "Kettle");
        s.threads
            .send_on(&thread, "Kettle", "Tea is ready.", None)
            .unwrap();
    });
    d.until("posted", |d| {
        d.record(GENERAL).is_some_and(|c| c.mirrored_through == 2)
    })
    .await;
    let agent = d.discord.seen()[0].user_agent.clone();
    assert_eq!(
        d.discord.seen(),
        [
            post(GENERAL, "Hello, zenworld.", &agent),
            post(GENERAL, "> **Kettle:** Tea is ready.", &agent),
        ]
    );
    d.settle().await;
    assert_eq!(d.discord.seen().len(), 2, "a post was made twice");

    d.rt.beat(&d.hosted);
    let records = world_store::read(&d.a.npcs.substrate(), WORLD);
    assert_eq!(
        records
            .get(&format!("discord/channel/{}/{GENERAL}", d.zen))
            .and_then(|r| r["mirrored_through"].as_u64()),
        Some(2)
    );
}

/// **A post Discord forbids closes the thread**, so the character stops
/// being offered it, and it is not tried again.
#[tokio::test]
async fn a_403_closes_the_thread_and_is_not_retried() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    d.discord.reply(
        403,
        json!({ "message": "Missing Permissions", "code": 50013 }),
    );
    d.hosted.with_sim(|s| {
        s.threads
            .send("Zen", "zenworld #general", "Can anyone hear me?")
            .unwrap();
    });
    d.until("closed", |d| {
        d.channel_thread(GENERAL).is_some_and(|t| !t.open)
    })
    .await;
    d.settle().await;
    assert_eq!(d.posts().len(), 1, "a forbidden post was tried again");
    assert_eq!(
        d.hosted.sim(|s| s.threads.names_for("Zen")),
        ["zenworld #lore"]
    );
    assert_eq!(d.a.bots.online().await, [d.zen], "a 403 refused the bot");
}

/// **A removal is told and closes the server's threads**; an outage is
/// neither.
#[tokio::test]
async fn a_removal_is_told_and_closes_the_servers_threads() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    d.told();
    d.gateway.event(
        "GUILD_DELETE",
        json!({ "id": GUILD.to_string(), "unavailable": true }),
    );
    d.settle().await;
    assert!(
        d.channel_thread(GENERAL).unwrap().open,
        "an outage closed it"
    );
    assert_eq!(d.notices(), Vec::<String>::new());

    d.gateway
        .event("GUILD_DELETE", json!({ "id": GUILD.to_string() }));
    d.until("left", |d| {
        d.hosted.servers(|s| s.server(d.zen, GUILD)).is_none()
    })
    .await;
    assert_eq!(
        d.notices(),
        ["You were removed from the Discord server zenworld."]
    );
    for channel in [GENERAL, LORE] {
        assert!(!d.channel_thread(channel).unwrap().open);
    }
    assert!(d.hosted.sim(|s| s.threads.names_for("Zen")).is_empty());
}

/// **`READY` against the records**: a recorded server it does not name was
/// left while offline, and is told; a server with no record is a join, and
/// is told once its `GUILD_CREATE` comes; a recorded one replayed is told
/// nothing, and takes its threads up again.
#[tokio::test]
async fn ready_tells_of_the_servers_left_and_joined_while_offline() {
    let d = daemon(vec![Script::Serve]).await;
    let (gone, home) = d.hosted.with_sim(|s| {
        (
            s.threads
                .open_standing("oldplace #chat", &["Zen".to_string()]),
            s.threads.open_standing("home #hall", &["Zen".to_string()]),
        )
    });
    d.hosted.with_servers(|s| {
        for (guild, name, channel, thread) in
            [(700, "oldplace", 710, &gone), (800, "home", 810, &home)]
        {
            s.put_server(d.zen, guild, Server { name: name.into() });
            s.put_channel(
                d.zen,
                channel,
                Channel {
                    guild_id: guild,
                    thread: thread.clone(),
                    mirrored_through: 0,
                    last_seen_discord: 900,
                },
            );
        }
    });
    d.gateway.in_guilds(&[GUILD, 800]);
    d.listen();
    d.online().await;
    assert_eq!(
        d.notices(),
        ["You were removed from the Discord server oldplace."]
    );
    assert!(!d.channel_thread(710).unwrap().open);

    d.gateway
        .event("GUILD_CREATE", guild(800, "home", &[(810, 0, "hall")]));
    d.gateway.event("GUILD_CREATE", zenworld());
    d.until("replayed", |d| d.record(LORE).is_some()).await;
    assert_eq!(
        d.notices(),
        [
            "Someone added you to the Discord server zenworld. You can talk in its channels: \
          zenworld #general, #lore"
        ]
    );
    let hall = d.channel_thread(810).unwrap();
    assert_eq!((hall.id, hall.open), (home, true));
}

/// **A renamed server or channel renames its threads, and a channel
/// described describes its thread**; a channel made is a thread, and one
/// deleted is closed and forgotten.
#[tokio::test]
async fn renames_rename_the_threads_and_channels_come_and_go() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    let general = d.record(GENERAL).unwrap().thread;
    let lore = d.record(LORE).unwrap().thread;

    d.gateway.event(
        "GUILD_UPDATE",
        json!({ "id": GUILD.to_string(), "name": "zen world" }),
    );
    d.gateway.event(
        "CHANNEL_UPDATE",
        json!({ "id": GENERAL.to_string(), "type": 0, "name": "lobby",
                "topic": "Say hello", "guild_id": GUILD.to_string() }),
    );
    d.gateway.event(
        "CHANNEL_CREATE",
        json!({ "id": "612", "type": 5, "name": "news", "guild_id": GUILD.to_string() }),
    );
    d.gateway.event(
        "CHANNEL_CREATE",
        json!({ "id": "613", "type": 2, "name": "Voice", "guild_id": GUILD.to_string() }),
    );
    d.gateway.event(
        "CHANNEL_DELETE",
        json!({ "id": LORE.to_string(), "type": 0, "guild_id": GUILD.to_string() }),
    );
    d.until("changed", |d| d.record(LORE).is_none()).await;

    let mut names = d.hosted.sim(|s| s.threads.names_for("Zen"));
    names.sort();
    assert_eq!(names, ["zen world #lobby", "zen world #news"]);
    assert_eq!(d.record(GENERAL).unwrap().thread, general, "a new thread");
    assert_eq!(
        d.hosted
            .sim(|s| s.threads.get(&general).unwrap().topic.clone()),
        Some("Say hello".to_string())
    );
    assert!(!d.hosted.sim(|s| s.threads.get(&lore).unwrap().open));
    assert_eq!(d.hosted.servers(|s| s.channel(d.zen, 613)), None);
    assert_eq!(
        d.hosted.servers(|s| s.server(d.zen, GUILD)),
        Some(Server {
            name: "zen world".into()
        })
    );
}

/// **What was written in a channel while no member ran the bot is heard
/// once it is online**: the gap fetch reads after the last message taken
/// in, one page, and hears it oldest first — the bot's own post dropped.
#[tokio::test]
async fn the_gap_fetch_hears_what_a_channel_missed() {
    let d = daemon(vec![Script::Serve]).await;
    let thread = d.hosted.with_sim(|s| {
        s.threads
            .open_standing("zenworld #general", &["Zen".to_string()])
    });
    d.hosted.with_servers(|s| {
        s.put_server(
            d.zen,
            GUILD,
            Server {
                name: "zenworld".into(),
            },
        );
        s.put_channel(
            d.zen,
            GENERAL,
            Channel {
                guild_id: GUILD,
                thread,
                mirrored_through: 0,
                last_seen_discord: 900,
            },
        );
    });
    d.gateway.in_guilds(&[GUILD]);
    // Read back over REST, a message names no server.
    let missed = |id: u64, author: u64, username: &str, content: &str| {
        let mut m = said(id, author, username, content);
        m.as_object_mut().unwrap().remove("guild_id");
        m
    };
    d.discord.history(
        GENERAL,
        vec![
            missed(903, KIT, "kit", "and again"),
            missed(902, BOT_USER, "Zen", "an old post of mine"),
            missed(901, KIT, "kit", "anyone?"),
        ],
    );
    d.until("heard", |d| d.heard(GENERAL).len() == 2).await;
    let agent = d.discord.seen()[0].user_agent.clone();
    assert_eq!(
        d.discord.seen()[0],
        as_bot(
            "GET",
            &format!("/channels/{GENERAL}/messages?after=900&limit=100"),
            Value::Null,
            &agent
        )
    );
    assert_eq!(
        d.heard(GENERAL),
        [
            ("kit (@kit)".to_string(), "anyone?".to_string()),
            ("kit (@kit)".to_string(), "and again".to_string()),
        ]
    );
    assert_eq!(d.record(GENERAL).unwrap().last_seen_discord, 903);
}

/// **The trace route answers the bot's state, shard, records and events**,
/// in exactly this shape: the channel heard, its thread and cursors, the
/// server, and every event from the shard opening to the message applied,
/// newest first. A character that does not exist is a 404.
#[tokio::test]
async fn the_trace_says_what_the_bridge_did() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    let id = later();
    d.gateway.dispatch(said(id, KIT, "kit", "anyone?"));
    d.until("heard", |d| d.heard(GENERAL).len() == 1).await;

    let res = get_trace(Shared(d.a.clone()), Wire(npc_id_wire(d.zen))).await;
    assert_eq!(res.status(), StatusCode::OK);
    let v: Value =
        serde_json::from_slice(&to_bytes(res.into_body(), 1 << 20).await.unwrap()).unwrap();
    let state = state::get(&d.a.npcs.substrate(), d.zen).unwrap();
    let general = d.record(GENERAL).unwrap();
    let lore = d.record(LORE).unwrap();
    let at = |i: usize| v["events"][i]["at_ms"].clone();
    assert_eq!(
        v,
        json!({
            "state": {
                "state": "online",
                "since_ms": state.since_ms,
                "world": WORLD,
                "connected_ms": 10,
                "member": null,
            },
            "shard": {
                "connected": true,
                "session": SESSION,
                "since_ms": v["shard"]["since_ms"],
                "last_event_ms": v["shard"]["last_event_ms"],
                "last_heartbeat_ack_ms": null,
                "member": null,
            },
            "dms": [],
            "channels": [
                {
                    "guild_id": GUILD, "channel_id": GENERAL, "thread": general.thread,
                    "name": "zenworld #general", "mirrored_through": 0,
                    "last_seen_discord": id, "thread_last_seq": 1, "unposted": 0, "open": true,
                },
                {
                    "guild_id": GUILD, "channel_id": LORE, "thread": lore.thread,
                    "name": "zenworld #lore", "mirrored_through": 0,
                    "last_seen_discord": lore.last_seen_discord, "thread_last_seq": null,
                    "unposted": 0, "open": true,
                },
            ],
            "servers": [{ "guild_id": GUILD, "name": "zenworld" }],
            "events": [
                {
                    "at_ms": at(0), "direction": "in", "kind": "channel", "message": id,
                    "channel": GENERAL, "author": KIT, "thread": general.thread, "seq": 1,
                    "outcome": "applied",
                },
                {
                    "at_ms": at(1), "direction": "in", "kind": "server", "guild": GUILD,
                    "outcome": "joined",
                },
                {
                    "at_ms": at(2), "direction": "shard", "kind": "ready", "author": BOT_USER,
                    "outcome": "online",
                },
                {
                    "at_ms": at(3), "direction": "shard", "kind": "opened",
                    "outcome": "connecting",
                },
            ],
        })
    );
    assert!(v["shard"]["since_ms"].as_u64().is_some());

    let missing = get_trace(Shared(d.a.clone()), Wire("zzzzzz".into())).await;
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
}

/// **A post that fails is in the trace with Discord's status and code**,
/// and one that is made with the id Discord gave it.
#[tokio::test]
async fn the_trace_keeps_each_post_and_each_failure() {
    let d = daemon(vec![Script::Serve]).await;
    d.joined().await;
    d.discord.reply(
        500,
        json!({ "message": "500: Internal Server Error", "code": 0 }),
    );
    d.hosted.with_sim(|s| {
        s.threads
            .send("Zen", "zenworld #general", "Hello?")
            .unwrap();
    });
    d.until("posted", |d| {
        d.record(GENERAL).is_some_and(|c| c.mirrored_through == 1)
    })
    .await;
    let thread = d.record(GENERAL).unwrap().thread;
    let outs: Vec<String> =
        d.a.bots
            .trace()
            .notes(d.zen)
            .into_iter()
            .filter(|n| n.direction == Direction::Out)
            .map(|n| n.line())
            .collect();
    assert_eq!(
        outs,
        [
            format!(
                "out channel message={FIRST_POSTED} channel={GENERAL} thread={thread} seq=1 \
                 parts=1: posted"
            ),
            format!("out channel channel={GENERAL} thread={thread} seq=1: failed 500/0"),
        ]
    );
}

/// **A `4014` is a bot waiting on its Message Content intent**: its state
/// and the console say so, and the shard asks again until Discord lets it
/// online.
#[tokio::test]
async fn a_4014_needs_the_content_intent_and_is_tried_again() {
    let d = daemon(vec![
        Script::Close(4014),
        Script::Close(4014),
        Script::Serve,
    ])
    .await;
    let shared = d.a.npcs.substrate();
    d.until("waiting", |d| {
        state::get(&d.a.npcs.substrate(), d.zen)
            .is_some_and(|s| s.state == State::NeedsContentIntent)
    })
    .await;
    assert_eq!(
        state::status(&shared, d.zen, &bot(&d.a, d.zen, 10)),
        state::Status::NeedsContentIntent
    );
    assert_eq!(d.a.bots.running().await, [d.zen], "the shard was stopped");
    d.online().await;
    assert_eq!(d.gateway.accepted(), 3);
    assert_eq!(state::get(&shared, d.zen).unwrap().state, State::Online);
}
