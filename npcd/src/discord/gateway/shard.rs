//! One bot's gateway connection: a task that keeps it open and hands on what
//! it hears.
//!
//! It connects, reads `HELLO`, and `IDENTIFY`s — or `RESUME`s the session it
//! had, at the URL `READY` gave for that — then heartbeats at the interval
//! `HELLO` asked for. `READY` and each `MESSAGE_CREATE` go out as an
//! [`Event`]. A connection that drops, is asked to reconnect, or misses a
//! heartbeat's acknowledgement is opened again after a wait that doubles to
//! [`MAX_WAIT`], resuming its session; a session Discord says is gone starts
//! afresh. A refused token (`4004`) and a request Discord will never grant
//! (`4010`–`4014`) end the task, saying which.
//!
//! The task never writes the world: it only hands on what it heard.

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use rand::Rng;
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{interval_at, sleep, timeout, Instant, MissedTickBehavior};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use super::protocol::{self, closed, op, Closed, Payload, CLOSE_TO_RESUME};
use crate::discord::message::DiscordMessage;

/// The longest a shard waits before connecting again.
pub const MAX_WAIT: Duration = Duration::from_secs(60);

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
}

/// What a shard hears.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Discord answered `READY`: the bot, as `user_id`, is online.
    Ready { user_id: u64 },
    /// A message the bot can see.
    Message(DiscordMessage),
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
        match shard.connection(&mut stopping).await {
            Ended::Stop => return,
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
            Ended::Reidentify => {
                shard.session = None;
                shard.seq = None;
            }
            Ended::Resume => {}
        }
        if shard.reached {
            wait = shard.config.retry;
        }
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
                    Err(code) => return ended(code),
                    Ok(None) => {}
                    Ok(Some(p)) => match p.op {
                        op::DISPATCH => self.dispatch(p),
                        op::HEARTBEAT => {
                            if send(&mut ws, &protocol::heartbeat(self.seq)).await.is_err() {
                                return Ended::Resume;
                            }
                        }
                        op::HEARTBEAT_ACK => acked = true,
                        op::RECONNECT => {
                            close_to_resume(&mut ws).await;
                            return Ended::Resume;
                        }
                        op::INVALID_SESSION => {
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
    /// handed on.
    fn dispatch(&mut self, p: Payload) {
        if let Some(s) = p.s {
            self.seq = Some(s);
        }
        let npc = self.config.npc;
        match p.t.as_deref() {
            Some("READY") => {
                self.reached = true;
                let id = p.d["session_id"].as_str().unwrap_or_default().to_string();
                let resume_url = p.d["resume_gateway_url"]
                    .as_str()
                    .unwrap_or(&self.config.gateway)
                    .to_string();
                self.session = Some(Session { id, resume_url });
                let user_id = p.d["user"]["id"]
                    .as_str()
                    .and_then(|id| id.parse().ok())
                    .unwrap_or(0);
                tracing::info!(npc, "discord: the bot is online");
                let _ = self.events.send((npc, Event::Ready { user_id }));
            }
            Some("RESUMED") => self.reached = true,
            Some("MESSAGE_CREATE") => match serde_json::from_value::<DiscordMessage>(p.d) {
                Ok(m) => {
                    let _ = self.events.send((npc, Event::Message(m)));
                }
                Err(e) => tracing::warn!(npc, "discord: a message that would not read: {e}"),
            },
            _ => {}
        }
    }
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

    const TOKEN: &str = "MTEx.bot.token";
    /// Long enough that no heartbeat falls inside a test that is not about
    /// one.
    const QUIET: u64 = 45_000;

    fn config(gateway: &str) -> Config {
        Config {
            npc: 42,
            gateway: gateway.to_string(),
            token: TOKEN.into(),
            intents: protocol::DIRECT_MESSAGES,
            activity: "in Life".into(),
            retry: Duration::from_millis(10),
        }
    }

    async fn next(events: &mut mpsc::UnboundedReceiver<(u64, Event)>) -> (u64, Event) {
        timeout(Duration::from_secs(10), events.recv())
            .await
            .expect("an event in time")
            .expect("the shard is still sending")
    }

    /// **`HELLO`, then `IDENTIFY` with only the DM intent, then `READY`**:
    /// the shard says the bot is online, as the user `READY` named.
    #[tokio::test]
    async fn a_shard_identifies_and_comes_online() {
        let gateway = fake::start(vec![Script::Serve], QUIET).await;
        let (tx, mut events) = mpsc::unbounded_channel();
        let shard = Shard::spawn(config(&gateway.url), tx);
        assert_eq!(
            next(&mut events).await,
            (42, Event::Ready { user_id: BOT_USER })
        );
        assert_eq!(
            gateway.received(),
            [protocol::identify(TOKEN, 4096, "in Life")]
        );
        shard.close().await;
        assert_eq!(gateway.accepted(), 1);
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
        assert_eq!(
            next(&mut events).await,
            (42, Event::Ready { user_id: BOT_USER })
        );
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
                protocol::identify(TOKEN, 4096, "in Life"),
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
                        bot: false
                    },
                    content: "hello".into(),
                    attachments: vec![],
                    webhook_id: None,
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
