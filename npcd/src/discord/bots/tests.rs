//! The bots against a fake Discord: its gateway and its REST API, each on a
//! port of the test's own, and a daemon with no engine hosting the vault, in
//! which Zen — [`OWNER`]'s character, with a bot — has a body.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use npc_map::world::Where;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::time::{sleep, timeout};
use web::auth::session::Identity;

use super::*;
use crate::accounts::{key_of, user_id_of, Accounts, DiscordUser};
use crate::authority::Role;
use crate::collections::Libraries;
use crate::discord::client::Client;
use crate::discord::dm::record::Dm;
use crate::discord::fake::{self, Fake, Mode, Seen, DM_CHANNEL};
use crate::discord::gateway::fake::{self as gateway_fake, FakeGateway, Script, BOT_USER};
use crate::discord::link::{code, index, linking};
use crate::discord::message::snowflake_at;
use crate::discord::store::{self, Bot};
use crate::engine::runtime::Runtime;
use crate::images::Images;
use crate::mind::Mind;
use crate::npcs::{self, Npcs};
use crate::projection::Source;
use crate::registry::Registry;
use crate::sim::phone::{Message, Origin};
use crate::world::Hosted;
use crate::world_state::store as world_store;

const ROOMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps");
const WORLD: &str = "creators-vault";
/// Zen's owner, who may DM it.
const OWNER: &str = "google-1";
/// Somebody else, who may not.
const STRANGER: &str = "google-2";
const TOKEN: &str = "MTEx.bot.token";
/// Wren's Discord user, and the DM channel Discord gave Wren and Zen's bot.
const WREN: u64 = 77;
const CHANNEL: u64 = 31;
/// What Wren is called in the world: the name the account speaks by.
const WREN_S: &str = "Wren S";

struct Daemon {
    a: Arc<Authored>,
    rt: Arc<Runtime>,
    discord: Fake,
    gateway: FakeGateway,
    zen: u64,
    hosted: Arc<Hosted>,
    _dir: TempDir,
}

fn identity(sub: &str) -> Identity {
    Identity {
        provider: "google".into(),
        sub: sub.into(),
        email: format!("{sub}@example.com"),
        name: match sub {
            OWNER => WREN_S.into(),
            _ => "Somebody Else".into(),
        },
        picture: String::new(),
        exp: 9_999_999_999,
    }
}

fn bot(a: &Authored, zen: u64, connected_ms: u64) -> Bot {
    Bot {
        application_id: "111".into(),
        user_id: BOT_USER.to_string(),
        username: "Zen".into(),
        token: a.discord_tokens.seal(zen, TOKEN),
        connected_ms,
    }
}

/// The daemon, its gateway playing `scripts`.
async fn daemon(scripts: Vec<Script>) -> Daemon {
    let dir = TempDir::new().unwrap();
    let base = dir.path();
    let npcs = Npcs::load(base).unwrap();
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
    let discord = fake::start(Mode::Ok).await;
    let gateway = gateway_fake::start(scripts, 45_000).await;
    authored.discord = Client::at(&discord.url);
    authored.bots = Bots::new(
        &gateway.url,
        "https://console.test/",
        Duration::from_millis(10),
    );
    let rt = Runtime::new(Mind::new(None), base).unwrap();
    rt.set_substrate(authored.npcs.substrate());
    rt.set_npcs(authored.npcs.clone());
    let a = authored.with_runtime(rt.clone());

    for sub in [OWNER, STRANGER] {
        a.accounts.write().await.upsert(&identity(sub), 1).unwrap();
    }
    let made = a
        .npcs
        .create(
            &identity(OWNER),
            &user_id_of(&identity(OWNER)),
            &json!({ "name": "Zen", "world_id": WORLD, "personality_id": "zen" }),
            1,
        )
        .unwrap();
    let zen = npcs::npc_id_of_wire(made["npc_id"].as_str().unwrap()).unwrap();
    let hosted = rt.host(WORLD, Path::new(ROOMS)).unwrap();
    rt.hold_world(WORLD, true);
    hosted.with(|w| {
        w.enter("zen", "Zen", Where::new("vault-casting", "green-room"))
            .unwrap()
    });
    rt.bodies.bind(zen, WORLD, "zen").unwrap();
    store::put(&a.npcs.substrate(), zen, &bot(&a, zen, 10)).unwrap();
    Daemon {
        a,
        rt,
        discord,
        gateway,
        zen,
        hosted,
        _dir: dir,
    }
}

impl Daemon {
    /// Beat until `done` says so.
    async fn until(&self, what: &str, done: impl Fn(&Daemon) -> bool) {
        timeout(Duration::from_secs(10), async {
            loop {
                self.a.bots.tick(&self.a).await;
                if done(self) {
                    return;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{what}: not in time"));
    }

    async fn online(&self) {
        let zen = self.zen;
        timeout(Duration::from_secs(10), async {
            while self.a.bots.online().await != [zen] {
                self.a.bots.tick(&self.a).await;
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the bot came online");
    }

    /// Beat a few times more, for what should not happen to have had its
    /// chance.
    async fn settle(&self) {
        for _ in 0..5 {
            self.a.bots.tick(&self.a).await;
            sleep(Duration::from_millis(20)).await;
        }
    }

    /// Link Discord user `discord_id` to account `sub`.
    async fn link(&self, sub: &str, discord_id: u64) {
        let user = DiscordUser {
            id: discord_id.to_string(),
            username: "wren".into(),
            linked_ms: 1,
        };
        linking::link(
            &*self.a.accounts.read().await,
            &self.a.npcs.substrate(),
            &key_of(&identity(sub)),
            &user,
        )
        .unwrap();
    }

    fn account(&self) -> String {
        key_of(&identity(OWNER))
    }

    /// Wren's thread with Zen.
    fn thread(&self) -> Vec<Message> {
        self.hosted.sim(|s| {
            s.threads
                .direct_between(WREN_S, "Zen")
                .map(|t| t.messages.clone())
                .unwrap_or_default()
        })
    }

    fn dm(&self) -> Option<Dm> {
        self.hosted.dms(|d| d.get(self.zen, &self.account()))
    }

    /// Zen says `text` to Wren.
    fn zen_says(&self, text: &str) {
        self.hosted.with_sim(|s| {
            s.threads.reach("Zen", WREN_S);
            s.threads.send("Zen", WREN_S, text).unwrap();
        });
    }

    /// Every post the fake was sent, as `(channel path, content)`.
    fn posts(&self) -> Vec<(String, String)> {
        self.discord
            .seen()
            .into_iter()
            .filter(|s| s.method == "POST" && s.path.ends_with("/messages"))
            .map(|s| {
                (
                    s.path,
                    s.body["content"].as_str().unwrap_or_default().into(),
                )
            })
            .collect()
    }
}

/// A DM to Zen's bot from `author`.
fn dm(id: u64, author: u64, content: &str) -> Value {
    json!({
        "id": id.to_string(),
        "channel_id": CHANNEL.to_string(),
        "author": { "id": author.to_string(), "username": "wren" },
        "content": content,
    })
}

fn as_bot(method: &str, path: &str, body: Value, agent: &str) -> Seen {
    Seen {
        method: method.into(),
        path: path.into(),
        authorization: format!("Bot {TOKEN}"),
        user_agent: agent.into(),
        body,
    }
}

fn post(channel: u64, content: &str, agent: &str) -> Seen {
    as_bot(
        "POST",
        &format!("/channels/{channel}/messages"),
        json!({ "content": content, "allowed_mentions": { "parse": [] } }),
        agent,
    )
}

// ── the shards follow the authority ──────────────────────────────────────

/// **A bot comes online on its world's authority**: one `IDENTIFY` with
/// only the DM intent, its status line naming the world, and its state
/// written online.
#[tokio::test]
async fn a_bot_comes_online_on_its_worlds_authority() {
    let d = daemon(vec![Script::Serve]).await;
    d.online().await;
    assert_eq!(
        d.gateway.received(),
        [protocol::identify(TOKEN, 4096, "in creators-vault")]
    );
    let held = state::get(&d.a.npcs.substrate(), d.zen).expect("a state");
    assert_eq!(
        (held.state, held.world.as_str(), held.connected_ms),
        (State::Online, WORLD, 10)
    );
    assert_eq!(
        state::status(&d.a.npcs.substrate(), d.zen, &bot(&d.a, d.zen, 10)),
        state::Status::Online {
            world: WORLD.into()
        }
    );
}

/// **A 4004 refuses the token**: the bot's state says so, its shard stops,
/// and it is not connected again — until a new token is.
#[tokio::test]
async fn a_refused_token_stops_the_bot_until_a_new_one_is_connected() {
    let d = daemon(vec![Script::Close(4004), Script::Serve]).await;
    let shared = d.a.npcs.substrate();
    d.until("refused", |d| {
        state::get(&d.a.npcs.substrate(), d.zen).is_some_and(|s| s.state == State::Refused)
    })
    .await;
    d.settle().await;
    assert_eq!(d.a.bots.running().await, Vec::<u64>::new());
    assert_eq!(d.gateway.accepted(), 1, "a refused token was tried again");

    store::put(&shared, d.zen, &bot(&d.a, d.zen, 11)).unwrap();
    d.online().await;
    assert_eq!(d.gateway.accepted(), 2);
    assert_eq!(state::get(&shared, d.zen).unwrap().state, State::Online);
}

/// **A lost authority closes the shard** and tombstones the bot's state; so
/// does the bot's record going.
#[tokio::test]
async fn a_lost_authority_or_a_removed_bot_closes_the_shard() {
    let d = daemon(vec![Script::Serve]).await;
    let shared = d.a.npcs.substrate();
    d.online().await;
    d.rt.standing.set_role(WORLD, Role::Replica, now_ms());
    d.a.bots.tick(&d.a).await;
    assert_eq!(d.a.bots.running().await, Vec::<u64>::new());
    assert_eq!(state::get(&shared, d.zen), None);

    d.rt.standing
        .set_role(WORLD, Role::Authority { epoch: 1 }, now_ms());
    d.online().await;
    store::remove(&shared, d.zen).unwrap();
    d.a.bots.tick(&d.a).await;
    assert_eq!(d.a.bots.running().await, Vec::<u64>::new());
    assert_eq!(state::get(&shared, d.zen), None);
}

// ── who is heard ─────────────────────────────────────────────────────────

/// **An unlinked author is answered once**, with where to link, and then
/// not at all; nothing of theirs enters the world.
#[tokio::test]
async fn an_unlinked_author_is_answered_once() {
    let d = daemon(vec![Script::Serve]).await;
    d.online().await;
    d.gateway.dispatch(dm(901, WREN, "hello?"));
    d.gateway.dispatch(dm(902, WREN, "hello??"));
    d.until("answered", |d| !d.posts().is_empty()).await;
    d.settle().await;
    let agent = d.discord.seen()[0].user_agent.clone();
    assert_eq!(
        d.discord.seen(),
        [post(
            CHANNEL,
            "I don't know you yet. Open https://console.test/#/profile, choose Link Discord, \
             and send me the code.",
            &agent
        )]
    );
    assert_eq!(d.thread(), []);
}

/// **A DM that is exactly a live code links its sender**, and says who they
/// are here; the code is spent.
#[tokio::test]
async fn a_link_code_links_its_sender() {
    let d = daemon(vec![Script::Serve]).await;
    let shared = d.a.npcs.substrate();
    let (minted, _) = code::mint(&shared, &d.account(), now_ms()).unwrap();
    d.online().await;
    d.gateway.dispatch(dm(901, WREN, &format!(" {minted} ")));
    d.until("linked", |d| !d.posts().is_empty()).await;
    assert_eq!(index::account_of(&shared, "77"), Some(d.account()));
    let linked = d.a.accounts.read().await.discord(&d.account()).unwrap();
    assert_eq!(
        (linked.id.as_str(), linked.username.as_str()),
        ("77", "wren")
    );
    assert_eq!(
        d.posts(),
        [(
            format!("/channels/{CHANNEL}/messages"),
            "Linked — you are Wren S here.".to_string()
        )]
    );
    assert_eq!(shared.custom_object(&code::key(&minted)), None);
    assert_eq!(d.thread(), [], "a code is not a message to the character");
}

/// **A linked person who may not see the character is told once**, and
/// nothing enters the world.
#[tokio::test]
async fn a_person_who_may_not_see_the_character_is_refused_once() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(STRANGER, 88).await;
    d.online().await;
    d.gateway.dispatch(dm(901, 88, "let me in"));
    d.gateway.dispatch(dm(902, 88, "please"));
    d.until("refused", |d| !d.posts().is_empty()).await;
    d.settle().await;
    assert_eq!(
        d.posts(),
        [(
            format!("/channels/{CHANNEL}/messages"),
            "I can't talk with you".to_string()
        )]
    );
    let stranger = d.hosted.sim(|s| {
        s.threads
            .direct_between("Somebody Else", "Zen")
            .map(|t| t.messages.len())
    });
    assert_eq!(stranger, None);
}

/// **An echo is not heard**: the bot's own post, another bot's, a webhook's,
/// and anything written in a server.
#[tokio::test]
async fn echoes_and_server_messages_are_not_heard() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    d.gateway.dispatch(dm(901, BOT_USER, "my own post"));
    let mut another_bot = dm(902, 99, "beep");
    another_bot["author"]["bot"] = json!(true);
    d.gateway.dispatch(another_bot);
    let mut hook = dm(903, WREN, "hooked");
    hook["webhook_id"] = json!("5");
    d.gateway.dispatch(hook);
    let mut server = dm(904, WREN, "in a server");
    server["guild_id"] = json!("6");
    d.gateway.dispatch(server);
    d.gateway.dispatch(dm(905, WREN, "the real one"));
    d.until("heard", |d| !d.thread().is_empty()).await;
    d.settle().await;
    assert_eq!(
        d.thread()
            .iter()
            .map(|m| m.intent.as_str())
            .collect::<Vec<_>>(),
        ["the real one"]
    );
    assert_eq!(d.discord.seen(), []);
}

// ── both ways ────────────────────────────────────────────────────────────

/// **End to end**: a DM arrives and is on the character's thread with its
/// origin, and not posted back; the character's answer, from the world, is
/// posted to the DM on the next beat, and the DM's record goes with the
/// world's records.
#[tokio::test]
async fn a_dm_is_heard_in_the_world_and_answered_on_the_next_beat() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    d.gateway.dispatch(dm(901, WREN, "are you there?"));
    d.until("heard", |d| !d.thread().is_empty()).await;

    let heard = d.thread();
    assert_eq!(
        (
            heard[0].from.as_str(),
            heard[0].intent.as_str(),
            heard[0].seq,
            heard[0].origin
        ),
        (
            WREN_S,
            "are you there?",
            1,
            Some(Origin::Discord {
                message: 901,
                channel: CHANNEL
            })
        )
    );
    assert_eq!(
        d.dm(),
        Some(Dm {
            channel_id: CHANNEL,
            user_id: WREN,
            mirrored_through: 1,
            last_seen_discord: 901,
        })
    );
    d.settle().await;
    assert_eq!(
        d.discord.seen(),
        [],
        "a message from Discord was posted back"
    );

    d.zen_says("I am. What do you need?");
    assert_eq!(d.thread()[1].origin, None);
    d.until("posted", |d| {
        d.dm().is_some_and(|dm| dm.mirrored_through == 2)
    })
    .await;
    let agent = d.discord.seen()[0].user_agent.clone();
    assert_eq!(
        d.discord.seen(),
        [post(CHANNEL, "I am. What do you need?", &agent)]
    );

    // The record is the world's, written with the rest of it.
    d.rt.beat(&d.hosted);
    let records = world_store::read(&d.a.npcs.substrate(), WORLD);
    assert_eq!(
        records.get(&format!("discord/dm/{}/{}", d.zen, d.account())),
        Some(&json!({
            "channel_id": CHANNEL,
            "user_id": WREN,
            "mirrored_through": 2,
            "last_seen_discord": 901,
        }))
    );
}

/// The same Discord message twice is heard once.
#[tokio::test]
async fn a_dm_already_heard_is_not_heard_again() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    d.gateway.dispatch(dm(901, WREN, "once"));
    d.gateway.dispatch(dm(901, WREN, "once"));
    d.until("heard", |d| !d.thread().is_empty()).await;
    d.settle().await;
    assert_eq!(d.thread().len(), 1);
}

/// **A 429 waits and the posts go in order**: the post tried again after
/// its wait, then the next.
#[tokio::test]
async fn a_429_is_waited_out_and_the_posts_stay_in_order() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    d.gateway.dispatch(dm(901, WREN, "hi"));
    d.until("heard", |d| d.dm().is_some()).await;
    d.discord.reply(
        429,
        json!({ "message": "You are being rate limited.", "retry_after": 0.05, "global": false }),
    );
    d.zen_says("one");
    d.zen_says("two");
    d.until("posted", |d| {
        d.dm().is_some_and(|dm| dm.mirrored_through == 3)
    })
    .await;
    assert_eq!(
        d.posts()
            .into_iter()
            .map(|(_, content)| content)
            .collect::<Vec<_>>(),
        ["one", "one", "two"]
    );
}

/// **A 500 leaves the cursor where it was**, and the next beat posts it.
#[tokio::test]
async fn a_500_is_posted_again_on_the_next_beat() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    d.gateway.dispatch(dm(901, WREN, "hi"));
    d.until("heard", |d| d.dm().is_some()).await;
    d.discord.reply(
        500,
        json!({ "message": "500: Internal Server Error", "code": 0 }),
    );
    d.zen_says("one");
    d.a.bots.tick(&d.a).await;
    assert_eq!(
        d.dm().unwrap().mirrored_through,
        1,
        "the cursor moved past a failure"
    );
    assert_eq!(d.posts().len(), 1);
    d.a.bots.tick(&d.a).await;
    assert_eq!(d.dm().unwrap().mirrored_through, 2);
    assert_eq!(
        d.posts()
            .into_iter()
            .map(|(_, content)| content)
            .collect::<Vec<_>>(),
        ["one", "one"]
    );
}

/// **A 401 on a post is a refused token**: the shard stops and the state
/// says so.
#[tokio::test]
async fn a_401_on_a_post_refuses_the_bot() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    d.gateway.dispatch(dm(901, WREN, "hi"));
    d.until("heard", |d| d.dm().is_some()).await;
    d.discord
        .reply(401, json!({ "message": "401: Unauthorized", "code": 0 }));
    d.zen_says("one");
    d.a.bots.tick(&d.a).await;
    assert_eq!(d.a.bots.running().await, Vec::<u64>::new());
    assert_eq!(
        state::get(&d.a.npcs.substrate(), d.zen).map(|s| s.state),
        Some(State::Refused)
    );
}

/// **A linked person the character writes to first gets a DM opened**, and
/// what the character wrote posted there.
#[tokio::test]
async fn the_character_writing_first_opens_a_dm() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    let before = snowflake_at(now_ms());
    d.zen_says("Are you around?");
    d.until("posted", |d| {
        d.dm().is_some_and(|dm| dm.mirrored_through == 1)
    })
    .await;
    let agent = d.discord.seen()[0].user_agent.clone();
    assert_eq!(
        d.discord.seen(),
        [
            as_bot(
                "POST",
                "/users/@me/channels",
                json!({ "recipient_id": "77" }),
                &agent
            ),
            post(DM_CHANNEL, "Are you around?", &agent),
        ]
    );
    let opened = d.dm().unwrap();
    assert_eq!((opened.channel_id, opened.user_id), (DM_CHANNEL, WREN));
    assert!(
        opened.last_seen_discord >= before,
        "{}",
        opened.last_seen_discord
    );
}

/// **No shared server, no DM**: the message stays unposted, and is not
/// asked about again until the character writes again.
#[tokio::test]
async fn with_no_shared_server_the_message_stays_unposted() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    d.online().await;
    d.discord.reply(
        400,
        json!({ "message": "Cannot send messages to this user", "code": 50278 }),
    );
    d.zen_says("Are you around?");
    d.a.bots.tick(&d.a).await;
    d.settle().await;
    assert_eq!(d.dm(), None);
    assert_eq!(d.discord.seen().len(), 1);
    d.zen_says("Hello?");
    d.until("posted", |d| {
        d.dm().is_some_and(|dm| dm.mirrored_through == 2)
    })
    .await;
    assert_eq!(
        d.posts()
            .into_iter()
            .map(|(_, content)| content)
            .collect::<Vec<_>>(),
        ["Are you around?", "Hello?"]
    );
}

/// **A DM written while no member ran the bot is heard once it is online**:
/// the gap fetch reads after the last message taken in, and hears what is
/// there oldest first — the bot's own post among it dropped.
#[tokio::test]
async fn the_gap_fetch_hears_a_missed_dm() {
    let d = daemon(vec![Script::Serve]).await;
    d.link(OWNER, WREN).await;
    let account = d.account();
    d.hosted.with_dms(|dms| {
        dms.put(
            d.zen,
            &account,
            Dm {
                channel_id: CHANNEL,
                user_id: WREN,
                mirrored_through: 0,
                last_seen_discord: 900,
            },
        )
    });
    d.discord.history(
        CHANNEL,
        vec![
            dm(903, WREN, "and again"),
            dm(902, BOT_USER, "an old post of mine"),
            dm(901, WREN, "missed you"),
        ],
    );
    d.until("heard", |d| d.thread().len() == 2).await;
    let agent = d.discord.seen()[0].user_agent.clone();
    assert_eq!(
        d.discord.seen()[0],
        as_bot(
            "GET",
            &format!("/channels/{CHANNEL}/messages?after=900&limit=100"),
            Value::Null,
            &agent
        )
    );
    assert_eq!(
        d.thread()
            .iter()
            .map(|m| (m.intent.as_str(), m.origin))
            .collect::<Vec<_>>(),
        [
            (
                "missed you",
                Some(Origin::Discord {
                    message: 901,
                    channel: CHANNEL
                })
            ),
            (
                "and again",
                Some(Origin::Discord {
                    message: 903,
                    channel: CHANNEL
                })
            ),
        ]
    );
    assert_eq!(d.dm().unwrap().last_seen_discord, 903);
}
