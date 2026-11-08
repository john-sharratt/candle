//! The house's asks over the gateway's bus (`docs/house_modes.md` §3). The
//! mode switch asks [`TOPIC`] to open or close rekordbox, naming an `id`, and
//! `zenj` answers on [`reply_topic`] of that id: `{ok, line}`. It subscribes
//! over the gateway's `/gateway/events` WebSocket as the service `zenj`, with
//! its key. With `zenj` not running nothing is subscribed, and the gateway
//! tells the asker at once that nothing listens — the house is never held
//! waiting on a DJ that is not there.

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::Message;

use crate::rekordbox::control::{Act, Rekordbox};

/// What the house asks rekordbox on.
pub const TOPIC: &str = "dj/rekordbox";

/// How long a dropped socket waits before connecting again.
const AGAIN: Duration = Duration::from_secs(5);

/// Where the answer to ask `id` is published.
pub fn reply_topic(id: &str) -> String {
    format!("{TOPIC}/result/{id}")
}

/// The frame that subscribes to the asks.
pub fn subscribe_frame() -> String {
    json!({"op": "subscribe", "topics": [TOPIC]}).to_string()
}

/// One ask as a frame from the bus carried it: its id, and what it asks or
/// why it asks nothing `zenj` does. `None` for a frame that is no ask — a
/// refusal, a message on another topic, one with no id to answer.
pub fn asked(frame: &str) -> Option<(String, Result<Act, String>)> {
    let f: Value = serde_json::from_str(frame).ok()?;
    if f["op"] != "message" || f["topic"] != TOPIC {
        return None;
    }
    let p = &f["payload"];
    let id = p["id"].as_str().filter(|id| !id.is_empty())?.to_string();
    let force = p["force"].as_bool().unwrap_or(false);
    let act = p["act"]
        .as_str()
        .and_then(|a| Act::parse(a, force))
        .ok_or_else(|| format!("`{}` is neither open nor close", p["act"]));
    Some((id, act))
}

/// The frame answering ask `id` with `done`.
pub fn answer_frame(id: &str, done: &Result<String, String>) -> String {
    let payload = match done {
        Ok(line) => json!({"ok": true, "line": line}),
        Err(why) => json!({"ok": false, "line": why}),
    };
    json!({"op": "publish", "topic": reply_topic(id), "payload": payload}).to_string()
}

/// Where the bus is, and how `zenj` is admitted to it.
#[derive(Clone)]
pub struct Bus {
    /// `ws://127.0.0.1/gateway/events`.
    pub url: String,
    /// The site named in `Host`.
    pub host: String,
    /// The service key.
    pub key: String,
}

impl Bus {
    /// The bus at `gateway` (`http://127.0.0.1`).
    pub fn at(gateway: &str, host: &str, key: &str) -> Bus {
        let base = gateway.trim_end_matches('/');
        let ws = match base.strip_prefix("https://") {
            Some(rest) => format!("wss://{rest}"),
            None => format!("ws://{}", base.trim_start_matches("http://")),
        };
        Bus {
            url: format!("{ws}/gateway/events"),
            host: host.to_string(),
            key: key.to_string(),
        }
    }

    /// Answer the house's asks for as long as `zenj` runs, connecting again
    /// whenever the socket drops.
    pub async fn serve(self, rekordbox: Arc<Rekordbox>) {
        loop {
            if let Err(e) = self.once(&rekordbox).await {
                tracing::warn!("bus: {e}; connecting again in {} s", AGAIN.as_secs());
            }
            tokio::time::sleep(AGAIN).await;
        }
    }

    async fn once(&self, rekordbox: &Arc<Rekordbox>) -> Result<(), String> {
        let mut req = self
            .url
            .as_str()
            .into_client_request()
            .map_err(|e| format!("{}: {e}", self.url))?;
        let headers = req.headers_mut();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", self.key)).map_err(|e| e.to_string())?,
        );
        headers.insert(
            "host",
            HeaderValue::from_str(&self.host).map_err(|e| e.to_string())?,
        );
        let (socket, _) = connect_async(req)
            .await
            .map_err(|e| format!("{}: {e}", self.url))?;
        let (mut up, mut down) = socket.split();
        up.send(Message::Text(subscribe_frame()))
            .await
            .map_err(|e| e.to_string())?;
        tracing::info!(topic = TOPIC, "bus: answering the house");
        let (answers, mut answered) = mpsc::channel::<String>(16);
        loop {
            tokio::select! {
                frame = down.next() => match frame {
                    Some(Ok(Message::Text(text))) => {
                        if text.contains("\"refused\"") {
                            return Err(format!("the gateway refused: {text}"));
                        }
                        let Some((id, act)) = asked(&text) else { continue };
                        let (rekordbox, answers) = (rekordbox.clone(), answers.clone());
                        tokio::spawn(async move {
                            let done = match act {
                                Ok(act) => rekordbox.act(act).await,
                                Err(why) => Err(why),
                            };
                            let _ = answers.send(answer_frame(&id, &done)).await;
                        });
                    }
                    Some(Ok(Message::Close(_))) | None => return Err("the socket closed".into()),
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(e.to_string()),
                },
                Some(frame) = answered.recv() => {
                    up.send(Message::Text(frame)).await.map_err(|e| e.to_string())?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **An ask on the topic is read with its id; anything else is no ask**,
    /// and an ask for something `zenj` does not do is answered as such.
    #[test]
    fn asks_are_read_from_the_bus_frames() {
        let frame = |payload: Value| {
            json!({"op": "message", "topic": TOPIC, "payload": payload, "at_ms": 1}).to_string()
        };
        assert_eq!(
            asked(&frame(json!({"id": "01J", "act": "open"}))),
            Some(("01J".into(), Ok(Act::Open)))
        );
        assert_eq!(
            asked(&frame(json!({"id": "02K", "act": "close", "force": true}))),
            Some(("02K".into(), Ok(Act::Close { force: true })))
        );
        assert_eq!(
            asked(&frame(json!({"id": "03L", "act": "dance"}))),
            Some(("03L".into(), Err("`\"dance\"` is neither open nor close".into())))
        );
        assert_eq!(asked(&frame(json!({"act": "open"}))), None, "no id to answer");
        assert_eq!(
            asked(&json!({"op": "message", "topic": "repo/x", "payload": {}}).to_string()),
            None
        );
        assert_eq!(asked("not json"), None);
    }

    /// **The answer goes to the ask's own reply topic**, as `{ok, line}`.
    #[test]
    fn an_answer_is_published_on_the_asks_reply_topic() {
        assert_eq!(
            answer_frame("01J", &Ok("rekordbox is open".into())),
            r#"{"op":"publish","topic":"dj/rekordbox/result/01J","payload":{"ok":true,"line":"rekordbox is open"}}"#
        );
        assert_eq!(
            answer_frame("01J", &Err("no".into())),
            r#"{"op":"publish","topic":"dj/rekordbox/result/01J","payload":{"ok":false,"line":"no"}}"#
        );
        assert_eq!(
            subscribe_frame(),
            r#"{"op":"subscribe","topics":["dj/rekordbox"]}"#
        );
    }

    #[test]
    fn the_bus_is_the_gateways_events_socket() {
        assert_eq!(
            Bus::at("http://127.0.0.1/", "code.tokera.com", "k").url,
            "ws://127.0.0.1/gateway/events"
        );
        assert_eq!(
            Bus::at("https://gw.example", "h", "k").url,
            "wss://gw.example/gateway/events"
        );
    }
}
