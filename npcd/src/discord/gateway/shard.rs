//! One bot's gateway connection: a task that keeps it open and hands on what
//! it hears.
//!
//! It connects, reads `HELLO`, and `IDENTIFY`s — or `RESUME`s the session it
//! had, at the URL `READY` gave for that — then heartbeats at the interval
//! `HELLO` asked for. `READY`, each `MESSAGE_CREATE`, and the server and
//! channel events (`GUILD_CREATE`, `GUILD_UPDATE`, `GUILD_DELETE`,
//! `CHANNEL_CREATE`/`UPDATE`/`DELETE`) go out as an [`Event`]. A connection
//! that drops, is asked to reconnect, or misses a heartbeat's acknowledgement
//! is opened again after a wait that doubles to [`MAX_WAIT`], resuming its
//! session; a session Discord says is gone starts afresh. A `4014` — the
//! Message Content intent not turned on — is said, and the shard asks again
//! after [`Config::content_retry`]. A refused token (`4004`) and a request
//! Discord will never grant (`4010`–`4013`) end the task, saying which.
//!
//! The task never writes the world: it only hands on what it heard.

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{interval_at, sleep, timeout, Instant, MissedTickBehavior};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use super::guild::{Guild, GuildChannel, GuildRef};
use super::protocol::{self, closed, op, Closed, Payload, CLOSE_TO_RESUME};
use crate::api::now_ms;
use crate::discord::message::{snowflake, DiscordMessage};
use crate::discord::trace::link::{update, SharedLink};

/// The longest a shard waits before connecting again.
pub const MAX_WAIT: Duration = Duration::from_secs(60);

/// How long a shard Discord closed with `4014` waits before asking again in
/// production: ten minutes, 144 `IDENTIFY`s a day, well inside the 1,000
/// Discord allows.
pub const CONTENT_RETRY: Duration = Duration::from_secs(600);

/// What one shard connects with.
#[derive(Clone)]
pub struct Config {
    /// The character whose bot this is: what each event is tagged with.
    pub npc: u64,
    /// The gateway's base, Discord's ([`protocol::PRODUCTION`]) or a test's.
    pub gateway: String,
    /// The bot token, opened.
    pub token: String,
    pub intents: u64,
    /// The bot's status line.
    pub activity: String,
    /// How long it waits before its first reconnect; each next waits twice
    /// as long, up to [`MAX_WAIT`].
    pub retry: Duration,
    /// How long it waits to ask again after a `4014`.
    pub content_retry: Duration,
    /// Where it keeps how its connection stands, for the trace.
    pub link: SharedLink,
}

/// What a shard hears.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Discord answered `READY`: the bot, as `user_id`, is online, in the
    /// servers `guilds` names.
    Ready { user_id: u64, guilds: Vec<u64> },
    /// A message the bot can see.
    Message(DiscordMessage),
    /// A server the bot is in, with its channels: one `READY` named, handed
    /// over after it, or one the bot was just added to.
    Guild(Guild),
    /// A server was renamed (`GUILD_UPDATE`).
    GuildRenamed { guild_id: u64, name: String },
    /// The bot was removed from a server — a `GUILD_DELETE` that is not an
    /// outage.
    Left { guild_id: u64 },
    /// A channel was made or changed (`CHANNEL_CREATE`, `CHANNEL_UPDATE`).
    Channel(GuildChannel),
    /// A channel was deleted.
    ChannelGone { channel_id: u64 },
    /// Discord closed the shard with `4014`: the application has not turned
    /// on the Message Content intent. The shard asks again later.
    NeedsContentIntent,
    /// Discord refused the token; the shard has stopped.
    Refused,
    /// The shard stopped for good, for this reason.
    Stopped(String),
}

/// A running shard.
pub struct Shard {
    stop: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl Shard {
    /// Start one, handing what it hears to `events` as `(npc, event)`.
    pub fn spawn(config: Config, events: mpsc::UnboundedSender<(u64, Event)>) -> Self {
        let (stop, stopping) = watch::channel(false);
        let task = tokio::spawn(run(config, events, stopping));
        Self { stop, task }
    }

    /// Close it: the connection is closed as Discord asks of a bot going
    /// offline, and the task ends.
    pub async fn close(self) {
        let _ = self.stop.send(true);
        let _ = self.task.await;
    }

    /// Whether its task has ended on its own.
    pub fn ended(&self) -> bool {
        self.task.is_finished()
    }
}

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// The session a shard resumes.
struct Session {
    id: String,
    resume_url: String,
}

/// How one connection ended.
enum Ended {
    /// The shard was told to stop.
    Stop,
    /// Discord refused the token.
    Refused,
    /// The Message Content intent is not turned on.
    NeedsContentIntent,
    /// Discord will never grant what the shard asks for.
    Fatal(String),
    /// The session is gone.
    Reidentify,
    /// The connection went; the session goes on.
    Resume,
}

/// What a shard keeps across its connections.
struct Running {
    config: Config,
    events: mpsc::UnboundedSender<(u64, Event)>,
    session: Option<Session>,
    seq: Option<u64>,
    /// Whether the connection now ending got as far as `READY` or `RESUMED`.
    reached: bool,
}

async fn run(
    config: Config,
    events: mpsc::UnboundedSender<(u64, Event)>,
    mut stopping: watch::Receiver<bool>,
) {
    let mut wait = config.retry;
    let mut shard = Running {
        config,
        events,
        session: None,
        seq: None,
        reached: false,
    };
    loop {
        if *stopping.borrow() {
            return;
        }
        shard.reached = false;
        let npc = shard.config.npc;
        let ended = shard.connection(&mut stopping).await;
        update(&shard.config.link, |l| {
            l.connected = false;
            l.since_ms = None;
        });
        match ended {
            Ended::Stop => {
                tracing::info!(npc, "discord: the bot's connection is closed");
                return;
            }
            Ended::Refused => {
                tracing::warn!(npc, "discord: the gateway refused the bot's token");
                let _ = shard.events.send((npc, Event::Refused));
                return;
            }
            Ended::Fatal(why) => {
                tracing::error!(npc, "discord: the gateway will not serve this bot: {why}");
                let _ = shard.events.send((npc, Event::Stopped(why)));
                return;
            }
            Ended::NeedsContentIntent => {
                tracing::warn!(
                    npc,
                    "discord: the bot's Message Content intent is not turned on; asking again later"
                );
                let _ = shard.events.send((npc, Event::NeedsContentIntent));
                shard.session = None;
                shard.seq = None;
                update(&shard.config.link, |l| l.session = None);
                tokio::select! {
                    _ = stopped(&mut stopping) => return,
                    _ = sleep(shard.config.content_retry) => {}
                }
                wait = shard.config.retry;
                continue;
            }
            Ended::Reidentify => {
                shard.session = None;
                shard.seq = None;
                update(&shard.config.link, |l| l.session = None);
            }
            Ended::Resume => {}
        }
        if shard.reached {
            wait = shard.config.retry;
        }
        tracing::info!(
            npc,
            resuming = shard.session.is_some(),
            "discord: reconnecting in {} ms",
            wait.as_millis()
        );
        tokio::select! {
            _ = stopped(&mut stopping) => return,
            _ = sleep(wait) => {}
        }
        wait = (wait * 2).min(MAX_WAIT);
    }
}

impl Running {
    /// One connection, from `HELLO` until it ends.
    async fn connection(&mut self, stopping: &mut watch::Receiver<bool>) -> Ended {
        let url = match &self.session {
            Some(s) => protocol::url(&s.resume_url),
            None => protocol::url(&self.config.gateway),
        };
        let opened = tokio::select! {
            _ = stopped(stopping) => return Ended::Stop,
            opened = open(self.config.npc, &url) => opened,
        };
        let (mut ws, every) = match opened {
            Ok(opened) => opened,
            Err(ended) => return ended,
        };
        let hello = match &self.session {
            Some(s) => protocol::resume(&self.config.token, &s.id, self.seq),
            None => protocol::identify(
                &self.config.token,
                self.config.intents,
                &self.config.activity,
            ),
        };
        if send(&mut ws, &hello).await.is_err() {
            return Ended::Resume;
        }
        // The first beat at a random point of the first interval, as Discord
        // asks, so bots that connected together do not beat together.
        let first = every.mul_f64(rand::rng().random_range(0.0..1.0));
        let mut beat = interval_at(Instant::now() + first, every);
        beat.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut acked = true;
        loop {
            tokio::select! {
                _ = stopped(stopping) => {
                    let _ = ws.close(Some(CloseFrame {
                        code: CloseCode::Normal,
                        reason: "going offline".into(),
                    })).await;
                    return Ended::Stop;
                }
                _ = beat.tick() => {
                    if !acked {
                        tracing::warn!(npc = self.config.npc, "discord: a heartbeat went unanswered; reconnecting");
                        close_to_resume(&mut ws).await;
                        return Ended::Resume;
                    }
                    acked = false;
                    if send(&mut ws, &protocol::heartbeat(self.seq)).await.is_err() {
                        return Ended::Resume;
                    }
                }
                frame = read(&mut ws) => match frame {
                    Err(code) => {
                        tracing::warn!(
                            npc = self.config.npc,
                            "discord: the gateway closed the connection (code {code:?})"
                        );
                        return ended(code);
                    }
                    Ok(None) => {}
                    Ok(Some(p)) => match p.op {
                        op::DISPATCH => self.dispatch(p),
                        op::HEARTBEAT => {
                            if send(&mut ws, &protocol::heartbeat(self.seq)).await.is_err() {
                                return Ended::Resume;
                            }
                        }
                        op::HEARTBEAT_ACK => {
                            acked = true;
                            update(&self.config.link, |l| l.last_heartbeat_ack_ms = Some(now_ms()));
                        }
                        op::RECONNECT => {
                            tracing::info!(npc = self.config.npc, "discord: the gateway asked for a reconnect");
                            close_to_resume(&mut ws).await;
                            return Ended::Resume;
                        }
                        op::INVALID_SESSION => {
                            tracing::info!(
                                npc = self.config.npc,
                                resumable = p.d.as_bool() == Some(true),
                                "discord: the gateway said the session is invalid"
                            );
                            close_to_resume(&mut ws).await;
                            return match p.d.as_bool() {
                                Some(true) => Ended::Resume,
                                _ => Ended::Reidentify,
                            };
                        }
                        _ => {}
                    },
                },
            }
        }
    }

    /// A dispatch: its seq kept, `READY` taken as the session, and a message
    /// or a server's or channel's change handed on.
    fn dispatch(&mut self, p: Payload) {
        if let Some(s) = p.s {
            self.seq = Some(s);
        }
        let npc = self.config.npc;
        let Some(t) = p.t.as_deref() else {
            return;
        };
        let now = now_ms();
        update(&self.config.link, |l| l.last_event_ms = Some(now));
        let event = match t {
            "READY" => {
                self.reached = true;
                let id = p.d["session_id"].as_str().unwrap_or_default().to_string();
                let resume_url = p.d["resume_gateway_url"]
                    .as_str()
                    .unwrap_or(&self.config.gateway)
                    .to_string();
                update(&self.config.link, |l| {
                    l.connected = true;
                    l.session = Some(id.clone());
                    l.since_ms = Some(now);
                });
                self.session = Some(Session { id, resume_url });
                let user_id = p.d["user"]["id"]
                    .as_str()
                    .and_then(|id| id.parse().ok())
                    .unwrap_or(0);
                let guilds = p.d["guilds"]
                    .as_array()
                    .map(|guilds| {
                        guilds
                            .iter()
                            .filter_map(|g| g["id"].as_str()?.parse().ok())
                            .collect()
                    })
                    .unwrap_or_default();
                tracing::info!(npc, "discord: the bot is online");
                Some(Event::Ready { user_id, guilds })
            }
            "RESUMED" => {
                self.reached = true;
                update(&self.config.link, |l| {
                    l.connected = true;
                    l.since_ms = Some(now);
                });
                tracing::info!(npc, "discord: the bot's session is resumed");
                None
            }
            "MESSAGE_CREATE" => data::<DiscordMessage>(npc, t, p.d).map(Event::Message),
            "GUILD_CREATE" => data::<Guild>(npc, t, p.d)
                .filter(|g| !g.unavailable)
                .map(Event::Guild),
            "GUILD_UPDATE" => data::<GuildRef>(npc, t, p.d).map(|g| Event::GuildRenamed {
                guild_id: g.id,
                name: g.name,
            }),
            // An outage is not a leave: the server comes back with a
            // `GUILD_CREATE` once Discord can serve it.
            "GUILD_DELETE" => data::<GuildRef>(npc, t, p.d)
                .filter(|g| !g.unavailable)
                .map(|g| Event::Left { guild_id: g.id }),
            "CHANNEL_CREATE" | "CHANNEL_UPDATE" => {
                data::<GuildChannel>(npc, t, p.d).map(Event::Channel)
            }
            "CHANNEL_DELETE" => {
                data::<Gone>(npc, t, p.d).map(|g| Event::ChannelGone { channel_id: g.id })
            }
            _ => None,
        };
        if let Some(event) = event {
            let _ = self.events.send((npc, event));
        }
    }
}

/// What a `CHANNEL_DELETE` names.
#[derive(Deserialize)]
struct Gone {
    #[serde(deserialize_with = "snowflake")]
    id: u64,
}

/// A dispatch's data read as `T`; one that does not read is logged and
/// gone past.
fn data<T: DeserializeOwned>(npc: u64, t: &str, d: Value) -> Option<T> {
    serde_json::from_value(d)
        .map_err(|e| tracing::warn!(npc, "discord: a {t} that would not read: {e}"))
        .ok()
}

/// Until the shard is told to stop — or its handle is gone, which is the
/// same.
async fn stopped(stopping: &mut watch::Receiver<bool>) {
    let _ = stopping.wait_for(|stop| *stop).await;
}

/// How long a gateway has to answer a connection with `HELLO`.
const HELLO_WITHIN: Duration = Duration::from_secs(20);

/// The heartbeat interval a `HELLO` that names none is taken to ask for.
const HEARTBEAT_UNSTATED: Duration = Duration::from_millis(41_250);

/// A connection to `url`, and the heartbeat interval its `HELLO` asked for.
async fn open(npc: u64, url: &str) -> Result<(Socket, Duration), Ended> {
    let hello = async {
        let (mut ws, _) = connect_async(url).await.map_err(|e| {
            tracing::warn!(npc, "discord: the gateway was not reached: {e}");
            Ended::Resume
        })?;
        match read(&mut ws).await {
            Ok(Some(p)) if p.op == op::HELLO => {
                let every = p.d["heartbeat_interval"]
                    .as_u64()
                    .map_or(HEARTBEAT_UNSTATED, Duration::from_millis);
                Ok((ws, every))
            }
            Ok(_) => Err(Ended::Resume),
            Err(code) => Err(ended(code)),
        }
    };
    timeout(HELLO_WITHIN, hello)
        .await
        .unwrap_or(Err(Ended::Resume))
}

/// What a connection closed with `code` — `None` for none — leaves the shard
/// to do.
fn ended(code: Option<u16>) -> Ended {
    match closed(code) {
        Closed::Refused => Ended::Refused,
        Closed::NeedsContentIntent => Ended::NeedsContentIntent,
        Closed::Fatal => Ended::Fatal(format!(
            "the gateway closed the connection with {}",
            code.unwrap_or_default()
        )),
        Closed::Reidentify => Ended::Reidentify,
        Closed::Resume => Ended::Resume,
    }
}

/// The next payload: `Ok(None)` for a frame that is not one, `Err` with the
/// close code (if any) for a connection that has ended.
async fn read(ws: &mut Socket) -> Result<Option<Payload>, Option<u16>> {
    match ws.next().await {
        None | Some(Err(_)) => Err(None),
        Some(Ok(Message::Close(frame))) => Err(frame.map(|f| u16::from(f.code))),
        Some(Ok(Message::Text(text))) => Ok(serde_json::from_str(&text).ok()),
        Some(Ok(_)) => Ok(None),
    }
}

async fn send(ws: &mut Socket, payload: &Value) -> Result<(), ()> {
    ws.send(Message::Text(payload.to_string()))
        .await
        .map_err(|_| ())
}

/// Close a connection so its session can be resumed.
async fn close_to_resume(ws: &mut Socket) {
    let _ = ws
        .close(Some(CloseFrame {
            code: CloseCode::from(CLOSE_TO_RESUME),
            reason: "reconnecting".into(),
        }))
        .await;
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokio::time::timeout;

    use super::*;
    use crate::discord::gateway::fake::{self, Script, BOT_USER, SESSION};
    use crate::discord::message::Author;
    use crate::discord::trace::link::read;

    const TOKEN: &str = "MTEx.bot.token";
    /// Long enough that no heartbeat falls inside a test that is not about
    /// one.
    const QUIET: u64 = 45_000;

    fn config(gateway: &str) -> Config {
        Config {
            npc: 42,
            gateway: gateway.to_string(),
            token: TOKEN.into(),
            intents: protocol::INTENTS,
            activity: "in Life".into(),
            retry: Duration::from_millis(10),
            content_retry: Duration::from_millis(30),
            link: SharedLink::default(),
        }
    }

    fn ready() -> Event {
        Event::Ready {
            user_id: BOT_USER,
            guilds: vec![],
        }
    }

    async fn next(events: &mut mpsc::UnboundedReceiver<(u64, Event)>) -> (u64, Event) {
        timeout(Duration::from_secs(10), events.recv())
            .await
            .expect("an event in time")
            .expect("the shard is still sending")
    }

    /// **`HELLO`, then `IDENTIFY` with the servers', messages', content and
    /// DM intents, then `READY`**: the shard says the bot is online, as the
    /// user `READY` named, in the servers it listed.
    #[tokio::test]
    async fn a_shard_identifies_and_comes_online() {
        let gateway = fake::start(vec![Script::Serve], QUIET).await;
        gateway.in_guilds(&[600, 700]);
        let (tx, mut events) = mpsc::unbounded_channel();
        let config = config(&gateway.url);
        let link = config.link.clone();
        let shard = Shard::spawn(config, tx);
        assert_eq!(
            next(&mut events).await,
            (
                42,
                Event::Ready {
                    user_id: BOT_USER,
                    guilds: vec![600, 700]
                }
            )
        );
        assert_eq!(
            gateway.received(),
            [protocol::identify(TOKEN, 37_377, "in Life")]
        );
        // The trace's view of the connection: open, on the session READY
        // opened, since the dispatch that said so.
        let seen = read(&link);
        assert!(seen.connected);
        assert_eq!(seen.session.as_deref(), Some(SESSION));
        assert!(seen.since_ms.is_some() && seen.since_ms == seen.last_event_ms);
        shard.close().await;
        assert_eq!(gateway.accepted(), 1);
        assert!(
            !read(&link).connected,
            "a closed shard still says it is connected"
        );
    }

    /// **A 4014 says the Message Content intent is off**, and the shard asks
    /// again after its wait — a fresh `IDENTIFY` — and comes online once it
    /// is on.
    #[tokio::test]
    async fn a_4014_close_says_so_and_asks_again_later() {
        let gateway = fake::start(vec![Script::Close(4014), Script::Serve], QUIET).await;
        let (tx, mut events) = mpsc::unbounded_channel();
        let shard = Shard::spawn(config(&gateway.url), tx);
        assert_eq!(next(&mut events).await, (42, Event::NeedsContentIntent));
        assert_eq!(next(&mut events).await, (42, ready()));
        assert_eq!(gateway.accepted(), 2);
        assert_eq!(
            gateway.received(),
            [
                protocol::identify(TOKEN, 37_377, "in Life"),
                protocol::identify(TOKEN, 37_377, "in Life"),
            ]
        );
        shard.close().await;
    }

    /// **The server and channel events are handed on**; a `GUILD_DELETE`
    /// that is an outage, and a `GUILD_CREATE` of a server unavailable, are
    /// not a leave or a join.
    #[tokio::test]
    async fn server_and_channel_events_are_handed_on() {
        let gateway = fake::start(vec![Script::Serve], QUIET).await;
        let (tx, mut events) = mpsc::unbounded_channel();
        let shard = Shard::spawn(config(&gateway.url), tx);
        next(&mut events).await;
        gateway.event("GUILD_CREATE", json!({"id": "601", "unavailable": true}));
        gateway.event(
            "GUILD_CREATE",
            json!({
                "id": "600", "name": "zenworld",
                "channels": [{"id": "610", "type": 0, "name": "general"}],
            }),
        );
        gateway.event("GUILD_UPDATE", json!({"id": "600", "name": "zen world"}));
        gateway.event(
            "CHANNEL_CREATE",
            json!({"id": "611", "type": 0, "name": "lore", "guild_id": "600"}),
        );
        gateway.event(
            "CHANNEL_DELETE",
            json!({"id": "611", "type": 0, "guild_id": "600"}),
        );
        gateway.event("GUILD_DELETE", json!({"id": "600", "unavailable": true}));
        gateway.event("GUILD_DELETE", json!({"id": "600"}));
        let mut heard = Vec::new();
        for _ in 0..5 {
            heard.push(next(&mut events).await.1);
        }
        assert_eq!(
            heard,
            [
                Event::Guild(Guild {
                    id: 600,
                    name: "zenworld".into(),
                    channels: vec![GuildChannel {
                        id: 610,
                        kind: 0,
                        name: "general".into(),
                        topic: None,
                        guild_id: None,
                    }],
                    unavailable: false,
                }),
                Event::GuildRenamed {
                    guild_id: 600,
                    name: "zen world".into()
                },
                Event::Channel(GuildChannel {
                    id: 611,
                    kind: 0,
                    name: "lore".into(),
                    topic: None,
                    guild_id: Some(600),
                }),
                Event::ChannelGone { channel_id: 611 },
                Event::Left { guild_id: 600 },
            ]
        );
        shard.close().await;
    }

    /// **A 4004 is a refused token**: said once, and the shard stops for good
    /// rather than spending identifies on a token Discord will not take.
    #[tokio::test]
    async fn a_4004_close_refuses_the_token_and_stops() {
        let gateway = fake::start(vec![Script::Close(4004)], QUIET).await;
        let (tx, mut events) = mpsc::unbounded_channel();
        let shard = Shard::spawn(config(&gateway.url), tx);
        assert_eq!(next(&mut events).await, (42, Event::Refused));
        timeout(Duration::from_secs(10), async {
            while !shard.ended() {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the shard stopped");
        sleep(Duration::from_millis(50)).await;
        assert_eq!(gateway.accepted(), 1, "it connected again");
    }

    /// **A dropped connection resumes its session** at the URL `READY` gave,
    /// naming the last dispatch it saw — and is not a second `READY`.
    #[tokio::test]
    async fn a_dropped_connection_resumes_its_session() {
        let gateway = fake::start(vec![Script::DropAfterReady, Script::Serve], QUIET).await;
        let (tx, mut events) = mpsc::unbounded_channel();
        let shard = Shard::spawn(config(&gateway.url), tx);
        assert_eq!(next(&mut events).await, (42, ready()));
        timeout(Duration::from_secs(10), async {
            while gateway.received().len() < 2 {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("it resumed");
        assert_eq!(
            gateway.received(),
            [
                protocol::identify(TOKEN, 37_377, "in Life"),
                protocol::resume(TOKEN, SESSION, Some(1)),
            ]
        );
        assert_eq!(gateway.accepted(), 2);
        // RESUMED is not READY again; the next event is what is dispatched.
        gateway.dispatch(json!({
            "id": "901", "channel_id": "31",
            "author": { "id": "77", "username": "wren" }, "content": "still there?",
        }));
        let (_, event) = next(&mut events).await;
        assert!(
            matches!(event, Event::Message(ref m) if m.id == 901),
            "{event:?}"
        );
        shard.close().await;
    }

    /// **A `MESSAGE_CREATE` is handed on as the message it carries.**
    #[tokio::test]
    async fn a_dispatched_message_is_handed_on() {
        let gateway = fake::start(vec![Script::Serve], QUIET).await;
        let (tx, mut events) = mpsc::unbounded_channel();
        let shard = Shard::spawn(config(&gateway.url), tx);
        next(&mut events).await;
        gateway.dispatch(json!({
            "id": "901", "channel_id": "31",
            "author": { "id": "77", "username": "wren" }, "content": "hello",
        }));
        assert_eq!(
            next(&mut events).await,
            (
                42,
                Event::Message(DiscordMessage {
                    id: 901,
                    channel_id: 31,
                    guild_id: None,
                    author: Author {
                        id: 77,
                        username: "wren".into(),
                        global_name: None,
                        bot: false
                    },
                    content: "hello".into(),
                    attachments: vec![],
                    webhook_id: None,
                    member: None,
                    mentions: vec![],
                    referenced_message: None,
                })
            )
        );
        shard.close().await;
    }

    /// **It heartbeats at the interval `HELLO` asked for**, naming the last
    /// dispatch it saw, and an answered beat keeps the connection.
    #[tokio::test]
    async fn a_shard_heartbeats_as_hello_asks() {
        let gateway = fake::start(vec![Script::Serve], 20).await;
        let (tx, mut events) = mpsc::unbounded_channel();
        let shard = Shard::spawn(config(&gateway.url), tx);
        next(&mut events).await;
        timeout(Duration::from_secs(10), async {
            while gateway.received().iter().filter(|p| p["op"] == 1).count() < 3 {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("it heartbeat");
        let beats: Vec<Value> = gateway
            .received()
            .into_iter()
            .filter(|p| p["op"] == 1)
            .collect();
        // The first may fall before `READY` is read, when there is no
        // dispatch yet to name.
        assert_eq!(
            beats.last(),
            Some(&protocol::heartbeat(Some(1))),
            "{beats:?}"
        );
        assert_eq!(
            gateway.accepted(),
            1,
            "an answered beat dropped the connection"
        );
        shard.close().await;
    }
}
