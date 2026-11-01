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

    /// Wick, a character of the owner's without a bot, with a body in the
    /// world; its id.
    fn wick(&self) -> u64 {
        let made = self
            .a
            .npcs
            .create(
                &identity(OWNER),
                &user_id_of(&identity(OWNER)),
                &json!({ "name": "Wick", "world_id": WORLD, "personality_id": "drone" }),
                2,
            )
            .unwrap();
        let wick = npcs::npc_id_of_wire(made["npc_id"].as_str().unwrap()).unwrap();
        self.hosted.with(|w| {
            w.enter("wick", "Wick", Where::new("vault-casting", "green-room"))
                .unwrap()
        });
        self.rt.bodies.bind(wick, WORLD, "wick").unwrap();
        wick
    }

    /// The reconcile, at once.
    async fn reconcile_now(&self) {
        self.a.bots.forget_reconciled(WORLD);
        self.a.bots.tick(&self.a).await;
    }

    /// What `thread` carries, as `(from, text, from_discord)`.
    fn said_on(&self, thread: &str) -> Vec<(String, String, bool)> {
        self.hosted.sim(|s| {
            s.threads.get(thread).map_or_else(Vec::new, |t| {
                t.messages
                    .iter()
                    .map(|m| (m.from.clone(), m.intent.clone(), m.origin.is_some()))
                    .collect()
            })
        })
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

/// The members-only channel a group gets: the first the fake makes after
/// `#the-channel`.
const GROUP: u64 = MADE + 1;

/// A message in channel or thread `channel` of the world's server.
fn posted_in(channel: u64, id: u64, author: u64, content: &str) -> Value {
    let mut m = posted_in_world(id, author, content);
    m["channel_id"] = json!(channel.to_string());
    m
}

/// **A group is a channel only its members see**: `@everyone` denied it,
/// the listener let in to mirror it, and each member on Discord — a linked
/// person — let see, read and write; a character without a bot is posted by
/// the webhook. A member's post on Discord is heard on the group's thread,
/// and anyone else's is dropped. A member who leaves is taken out, and a
/// group nobody remains on is archived — read-only, and posted to no more.
#[tokio::test]
async fn a_group_is_a_channel_only_its_members_see() {
    use crate::discord::world::members::{HISTORY, LISTENER, MEMBER, SEND, VIEW};

    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.link(CREATOR, ADA).await;
    d.wick();
    let sweep = d.hosted.with_sim(|s| {
        s.threads.open_group(
            "The Eastern Sweep",
            &["Zen".into(), "Wick".into(), WREN_S.into()],
        )
    });
    d.mirroring().await;
    d.until("the group opened", |d| {
        d.hosted.mirror(|m| m.thread(&sweep)).is_some()
    })
    .await;

    let made = d
        .discord
        .seen()
        .into_iter()
        .find(|s| {
            s.method == "POST"
                && s.path == "/guilds/600/channels"
                && s.body["name"] == "the-eastern-sweep"
        })
        .expect("the group's channel was made");
    assert_eq!(made.body["type"], json!(0));
    assert_eq!(
        d.discord.overwrites_of(GROUP),
        [
            (WREN, 1, MEMBER, 0),
            (BOT_USER, 1, LISTENER, 0),
            (GUILD, 0, 0, VIEW),
        ]
    );
    let mirrored = d.hosted.mirror(|m| m.thread(&sweep)).unwrap();
    assert_eq!((mirrored.channel_id, mirrored.kind), (GROUP, Kind::Group));

    d.hosted.with_sim(|s| {
        s.threads
            .send_on(&sweep, "Wick", "the east is clear", None)
            .unwrap()
    });
    d.until("posted", |d| {
        d.hooked()
            .contains(&("Wick".to_string(), "the east is clear".to_string()))
    })
    .await;

    let first = later();
    d.gateway
        .dispatch(posted_in(GROUP, first, WREN, "then we move at dawn"));
    d.gateway
        .dispatch(posted_in(GROUP, first + 1, ADA, "I am not on this"));
    d.until("the member heard", |d| d.said_on(&sweep).len() == 2)
        .await;
    d.settle().await;
    assert_eq!(
        d.said_on(&sweep),
        [
            ("Wick".to_string(), "the east is clear".to_string(), false),
            (WREN_S.to_string(), "then we move at dawn".to_string(), true),
        ]
    );

    // Wren leaves: taken out of the channel.
    d.hosted
        .with_sim(|s| s.threads.leave(WREN_S, "The Eastern Sweep").unwrap());
    d.reconcile_now().await;
    assert_eq!(
        d.discord.overwrites_of(GROUP),
        [(BOT_USER, 1, LISTENER, 0), (GUILD, 0, 0, VIEW)]
    );

    // Wren is let back in, and then everybody goes: archived, read-only.
    d.hosted.with_sim(|s| {
        s.threads
            .invite("Zen", "The Eastern Sweep", WREN_S)
            .unwrap();
    });
    d.reconcile_now().await;
    assert!(d
        .discord
        .overwrites_of(GROUP)
        .contains(&(WREN, 1, MEMBER, 0)));
    d.hosted.with_sim(|s| {
        for who in ["Zen", "Wick", WREN_S] {
            let _ = s.threads.leave(who, "The Eastern Sweep");
        }
    });
    d.reconcile_now().await;
    assert!(d.hosted.mirror(|m| m.thread(&sweep)).unwrap().archived);
    assert!(d
        .discord
        .overwrites_of(GROUP)
        .contains(&(WREN, 1, VIEW | HISTORY, SEND)));
}

/// **A person's DM with a character without a bot is a private thread**
/// under `#dms`: the channel found or made, everyone let see it and write in
/// none of it but its threads; a thread named for the two of them that the
/// person is added to. The character's lines are posted into it through
/// `#dms`'s webhook, and the person's are heard as a message to the
/// character.
#[tokio::test]
async fn a_dm_with_a_character_without_a_bot_is_a_private_thread() {
    use crate::discord::world::members::{HISTORY, LISTENER, SEND, VIEW};

    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.wick();
    d.hosted.with_sim(|s| {
        s.threads.reach(WREN_S, "Wick");
        s.threads
            .send(WREN_S, "Wick", "said before the server")
            .unwrap();
    });
    let dm = d.hosted.sim(|s| {
        s.threads
            .direct_between(WREN_S, "Wick")
            .map(|t| t.id.clone())
            .unwrap()
    });
    d.mirroring().await;
    d.until("the DM opened", |d| {
        d.hosted.mirror(|m| m.thread(&dm)).is_some()
    })
    .await;

    let dms = d.hosted.mirror(|m| m.parent("dms")).expect("#dms recorded");
    assert_eq!(
        d.discord.overwrites_of(dms),
        [(BOT_USER, 1, LISTENER, 0), (GUILD, 0, VIEW | HISTORY, SEND)]
    );
    let mirrored = d.hosted.mirror(|m| m.thread(&dm)).unwrap();
    assert_eq!(
        (mirrored.kind, mirrored.parent, mirrored.mirrored_through),
        (Kind::Dm, Some(dms), 1)
    );
    let opened: Vec<(String, Value)> = d
        .discord
        .seen()
        .into_iter()
        .filter(|s| s.path.contains("/threads") || s.path.contains("/thread-members/"))
        .map(|s| (format!("{} {}", s.method, s.path), s.body))
        .collect();
    assert_eq!(
        opened,
        [
            (
                format!("POST /channels/{dms}/threads"),
                json!({ "name": "Wick & Wren S", "type": 12, "invitable": false })
            ),
            (
                format!(
                    "PUT /channels/{}/thread-members/{WREN}",
                    mirrored.channel_id
                ),
                Value::Null
            ),
        ]
    );

    d.hosted
        .with_sim(|s| s.threads.send("Wick", WREN_S, "hello, Wren").unwrap());
    d.until("posted into the thread", |d| {
        d.discord.seen().iter().any(|s| {
            s.path
                == format!(
                    "/webhooks/{}/{HOOK_TOKEN}?wait=true&thread_id={}",
                    dms + HOOK_OFFSET,
                    mirrored.channel_id
                )
        })
    })
    .await;
    assert_eq!(
        d.hooked(),
        [("Wick".to_string(), "hello, Wren".to_string())],
        "what was said before the server was posted"
    );

    d.gateway.dispatch(posted_in(
        mirrored.channel_id,
        later(),
        WREN,
        "how are the hatchlings?",
    ));
    d.until("the person heard", |d| d.said_on(&dm).len() == 3)
        .await;
    assert_eq!(
        d.said_on(&dm).last(),
        Some(&(
            WREN_S.to_string(),
            "how are the hatchlings?".to_string(),
            true
        ))
    );
}

/// **Two characters' DM is a backstage thread their owner reads**: under
/// `#backstage`, which everyone may see and nobody may write in or its
/// threads, a thread named for the two with their owner added once; each
/// character's lines posted by the webhook — Zen's too, though it has a bot
/// — and anything written there dropped. A world whose server does not
/// mirror them opens none.
#[tokio::test]
async fn two_characters_dm_is_a_backstage_thread_their_owner_reads() {
    use crate::discord::world::members::{HISTORY, LISTENER, SEND, SEND_IN_THREADS, VIEW};

    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.wick();
    d.hosted.with_sim(|s| {
        s.threads.reach("Zen", "Wick");
    });
    let dm = d.hosted.sim(|s| {
        s.threads
            .direct_between("Zen", "Wick")
            .map(|t| t.id.clone())
            .unwrap()
    });
    d.mirroring().await;
    d.until("backstage opened", |d| {
        d.hosted.mirror(|m| m.thread(&dm)).is_some()
    })
    .await;

    let parent = d.hosted.mirror(|m| m.parent("backstage")).unwrap();
    assert_eq!(
        d.discord.overwrites_of(parent),
        [
            (BOT_USER, 1, LISTENER, 0),
            (GUILD, 0, VIEW | HISTORY, SEND | SEND_IN_THREADS),
        ]
    );
    let mirrored = d.hosted.mirror(|m| m.thread(&dm)).unwrap();
    assert_eq!(
        (mirrored.kind, mirrored.parent),
        (Kind::Backstage, Some(parent))
    );
    let added: Vec<String> = d
        .discord
        .seen()
        .into_iter()
        .filter(|s| s.path.contains("/thread-members/"))
        .map(|s| s.path)
        .collect();
    assert_eq!(
        added,
        [format!(
            "/channels/{}/thread-members/{WREN}",
            mirrored.channel_id
        )]
    );

    d.hosted
        .with_sim(|s| s.threads.send("Zen", "Wick", "report at dusk").unwrap());
    d.until("posted backstage", |d| {
        d.hooked()
            .contains(&("Zen".to_string(), "report at dusk".to_string()))
    })
    .await;
    d.gateway.dispatch(posted_in(
        mirrored.channel_id,
        later(),
        WREN,
        "a word from me",
    ));
    d.settle().await;
    assert_eq!(d.said_on(&dm).len(), 1, "a word was written backstage");

    let quiet = daemon(vec![Script::Serve]).await;
    quiet.link(OWNER, WREN).await;
    quiet.wick();
    quiet.hosted.with_sim(|s| {
        s.threads.reach("Zen", "Wick");
    });
    quiet.serve_world();
    quiet.hosted.with_mirror(|m| {
        let mut g = m.guild().cloned().unwrap();
        g.backstage = false;
        m.set_guild(Some(g));
    });
    quiet.online().await;
    quiet.settle().await;
    assert!(quiet.hosted.mirror(|m| m.parent("backstage")).is_none());
}

/// **The cursors go with the world** (§5.4): the member that held it wrote
/// them with the world's records, and the member that takes it next — the
/// records applied, its own listener online — posts on from where they
/// stopped, and what the first posted is not posted again.
#[tokio::test]
async fn the_cursors_carry_over_to_the_next_holder() {
    let first = daemon(vec![Script::Serve]).await;
    first.mirroring().await;
    first.on_channel("Wick", "posted by the first holder");
    first.until("posted", |d| d.hooked().len() == 1).await;
    first.rt.beat(&first.hosted);
    let records = world_store::read(&first.a.npcs.substrate(), WORLD);
    first.a.bots.close_all(&first.a).await;
    let thread = first.the_channel();
    let through = first
        .hosted
        .mirror(|m| m.thread(&thread))
        .unwrap()
        .mirrored_through;

    let next = daemon(vec![Script::Serve]).await;
    crate::world_state::apply(&next.hosted, &records);
    assert_eq!(
        next.hosted
            .mirror(|m| m.thread(&thread))
            .map(|m| m.mirrored_through),
        Some(through),
        "the cursor did not come with the records"
    );
    // The listener is the next member's own character there.
    next.hosted.with_mirror(|m| {
        let mut g = m.guild().cloned().unwrap();
        g.listener = next.zen;
        m.set_guild(Some(g));
    });
    next.online().await;
    next.on_channel("Wick", "posted by the next");
    next.until("posted on", |d| !d.hooked().is_empty()).await;
    next.settle().await;
    assert_eq!(
        next.hooked(),
        [("Wick".to_string(), "posted by the next".to_string())],
        "the next holder posted what the first had"
    );
}

/// **A bot shows typing while a person's DM waits on it**, and stops once
/// the character answers.
#[tokio::test]
async fn a_bot_shows_typing_while_a_dm_waits_on_it() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    let typing = |d: &Daemon| {
        d.discord
            .seen()
            .iter()
            .filter(|s| s.method == "POST" && s.path == format!("/channels/{CHANNEL}/typing"))
            .count()
    };
    d.gateway.dispatch(dm(later(), WREN, "are you there?"));
    d.until("typing", |d| typing(d) == 1).await;
    d.settle().await;
    assert_eq!(typing(&d), 1, "typing was asked again inside eight seconds");

    d.zen_says("I am here");
    d.until("answered", |d| {
        d.posts().iter().any(|(_, text)| text == "I am here")
    })
    .await;
    d.settle().await;
    assert_eq!(typing(&d), 1, "typing went on after the answer");
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
