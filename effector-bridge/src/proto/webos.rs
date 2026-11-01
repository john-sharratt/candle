//! The LG webOS TV protocol (SSAP): a WebSocket on port 3000 carrying JSON
//! requests addressed by `ssap://` URIs.
//!
//! The first connection registers: the TV shows an on-screen prompt, and once
//! accepted it returns a `client-key` to keep. Later connections present that
//! key and register silently. A request names an `ssap://` service and gets a
//! response correlated by id.

use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

const PORT: u16 = 3000;
/// How long to wait for an ordinary request's reply.
const ANSWER: Duration = Duration::from_secs(6);
/// How long to wait for the person to accept the on-screen pairing prompt.
const PAIR: Duration = Duration::from_secs(150);

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// The registration payload: the standard webOS remote manifest, with the
/// stored `client-key` added when one is held so the TV registers silently.
pub fn register_payload(client_key: Option<&str>) -> Value {
    let mut payload = json!({
        "forcePairing": false,
        "pairingType": "PROMPT",
        "manifest": {
            "manifestVersion": 1,
            "appVersion": "1.1",
            "signed": {
                "created": "20140509",
                "appId": "com.lge.test",
                "vendorId": "com.lge",
                "localizedAppNames": { "": "LG Remote App" },
                "localizedVendorNames": { "": "LG Electronics" },
                "permissions": [
                    "TEST_SECURE", "CONTROL_INPUT_TEXT", "CONTROL_MOUSE_AND_KEYBOARD",
                    "READ_INSTALLED_APPS", "READ_LGE_SDX", "READ_NOTIFICATIONS",
                    "SEARCH", "WRITE_SETTINGS", "WRITE_NOTIFICATION_ALERT",
                    "CONTROL_POWER", "READ_CURRENT_CHANNEL", "READ_RUNNING_APPS",
                    "READ_UPDATE_INFO", "UPDATE_FROM_REMOTE_APP", "READ_LGE_TV_INPUT_EVENTS",
                    "READ_TV_CURRENT_TIME"
                ],
                "serial": "2f930e2d2cfe083771f68e4fe7bb07"
            },
            "permissions": [
                "LAUNCH", "LAUNCH_WEBAPP", "APP_TO_APP", "CLOSE", "TEST_OPEN",
                "TEST_PROTECTED", "CONTROL_AUDIO", "CONTROL_DISPLAY",
                "CONTROL_INPUT_JOYSTICK", "CONTROL_INPUT_MEDIA_RECORDING",
                "CONTROL_INPUT_MEDIA_PLAYBACK", "CONTROL_INPUT_TV", "CONTROL_POWER",
                "READ_APP_STATUS", "READ_CURRENT_CHANNEL", "READ_INPUT_DEVICE_LIST",
                "READ_NETWORK_STATE", "READ_RUNNING_APPS", "READ_TV_CHANNEL_LIST",
                "WRITE_NOTIFICATION_TOAST", "READ_POWER_STATE", "READ_COUNTRY_INFO",
                "READ_SETTINGS", "CONTROL_TV_SCREEN", "CONTROL_TV_STANBY",
                "CONTROL_FAVORITE_GROUP", "CONTROL_BLUETOOTH", "CONTROL_TIMER_INFO",
                "READ_TV_CURRENT_TIME", "ADD_LAUNCHER_CHANNEL", "CONTROL_TV_POWER",
                "CONTROL_WOL", "READ_INSTALLED_APPS", "CONTROL_MOUSE_AND_KEYBOARD",
                "CONTROL_INPUT_TEXT", "READ_NOTIFICATIONS", "WRITE_SETTINGS"
            ],
            "signatures": [{
                "signatureVersion": 1,
                "signature": "eyJhbGdvcml0aG0iOiJSU0EtU0hBMjU2Iiwia2V5SWQiOiJ0ZXN0LXNpZ25pbmctY2VydCIsInNpZ25hdHVyZVZlcnNpb24iOjF9.hrVRgjCwXVvE2OOSpDZ58hR+59aFNwYDyjQgKk3auukd7pcegmE2CzPCa0bJ0ZsRAcKkCTJrWo5iDzNhMBWRyaMOv5zWSrthlf7G128qvIlpMT0YNY+n/FaOHE73uLrS/g7swl3/qH/BGFG2Hu4RlL48eb3lLKqTt2xKHdCs6Cd4RMfJPYnzgvI4BNrFUKsjkcu+WD4OO2A27Pq1n50cMchmcaXadJhGrOqH5YmHdOCj5NSHzJYrsW0HPlpuAx/ECMeIZYDh6RMqaFM2DXzdKX9NmmyqzJ3o/0lkk/N97gfVRLW5hA29yeAwaCViZNCP8iC9aO0q9fQojoa7NQnAtw=="
            }]
        }
    });
    if let Some(key) = client_key {
        payload["client-key"] = json!(key);
    }
    payload
}

/// An open SSAP connection to one webOS TV.
pub struct Remote {
    socket: Socket,
    client_key: Option<String>,
    next_id: u64,
}

impl Remote {
    /// Open the socket and register. With no key (or an unknown one) the TV
    /// shows an on-screen prompt and, once accepted, returns a `client-key` to
    /// keep; with a valid key it registers silently.
    pub async fn open(address: &str, client_key: Option<&str>) -> Result<Remote> {
        let url = format!("ws://{address}:{PORT}");
        let (socket, _) = timeout(ANSWER, connect_async(url))
            .await
            .map_err(|_| anyhow!("the TV did not answer in time"))??;
        let mut remote = Remote {
            socket,
            client_key: client_key.map(str::to_string),
            next_id: 1,
        };
        remote.register(client_key).await?;
        Ok(remote)
    }

    async fn register(&mut self, client_key: Option<&str>) -> Result<()> {
        let msg = json!({
            "type": "register",
            "id": "register_0",
            "payload": register_payload(client_key),
        });
        self.socket.send(Message::Text(msg.to_string())).await?;
        // The reply is a "response" that the prompt is up, then a "registered"
        // once accepted (or straight to "registered" with a valid key).
        loop {
            let frame = timeout(PAIR, self.socket.next())
                .await
                .map_err(|_| anyhow!("the pairing was not accepted in time — allow it on the TV"))?
                .ok_or_else(|| anyhow!("the TV closed the connection"))??;
            let Message::Text(text) = frame else { continue };
            let v: Value = serde_json::from_str(text.as_str()).unwrap_or_default();
            match v["type"].as_str() {
                Some("registered") => {
                    if let Some(k) = v["payload"]["client-key"].as_str() {
                        self.client_key = Some(k.to_string());
                    }
                    return Ok(());
                }
                Some("error") => {
                    bail!(
                        "the TV refused the pairing: {}",
                        v["error"].as_str().unwrap_or("rejected")
                    )
                }
                // A "response" with the prompt type just means the prompt is up.
                _ => continue,
            }
        }
    }

    /// The client key the TV granted, to keep for next time.
    pub fn client_key(&self) -> Option<&str> {
        self.client_key.as_deref()
    }

    /// Send an `ssap://` request and return its response payload.
    pub async fn request(&mut self, uri: &str, payload: Value) -> Result<Value> {
        let id = format!("req_{}", self.next_id);
        self.next_id += 1;
        let msg = json!({ "type": "request", "id": id, "uri": uri, "payload": payload });
        self.socket.send(Message::Text(msg.to_string())).await?;
        loop {
            let frame = timeout(ANSWER, self.socket.next())
                .await
                .map_err(|_| anyhow!("the TV went quiet"))?
                .ok_or_else(|| anyhow!("the TV closed the connection"))??;
            let Message::Text(text) = frame else { continue };
            let v: Value = serde_json::from_str(text.as_str()).unwrap_or_default();
            if v["id"].as_str() != Some(id.as_str()) {
                continue;
            }
            if v["type"].as_str() == Some("error") {
                bail!(
                    "the TV refused it: {}",
                    v["error"].as_str().unwrap_or("error")
                );
            }
            let payload = v.get("payload").cloned().unwrap_or_default();
            if payload["returnValue"].as_bool() == Some(false) {
                bail!(
                    "the TV refused it: {}",
                    payload["errorText"].as_str().unwrap_or("refused")
                );
            }
            return Ok(payload);
        }
    }

    /// Send a request that expects no meaningful payload back.
    pub async fn tell(&mut self, uri: &str, payload: Value) -> Result<()> {
        self.request(uri, payload).await.map(drop)
    }

    /// The URL of the TV's pointer-input socket — a second WebSocket that
    /// carries remote-control button presses (`HOME`, `UP`, `ENTER`, …).
    pub async fn pointer_socket(&mut self) -> Result<String> {
        let p = self
            .request(
                "ssap://com.webos.service.networkinput/getPointerInputSocket",
                json!({}),
            )
            .await?;
        p["socketPath"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("the TV gave no pointer socket"))
    }
}

/// Press one remote button by opening the pointer-input socket, sending it, and
/// closing — the socket speaks a tiny line protocol, not SSAP JSON.
pub async fn send_button(socket_path: &str, name: &str) -> Result<()> {
    let (mut socket, _) = timeout(ANSWER, connect_async(socket_path))
        .await
        .map_err(|_| anyhow!("the pointer socket did not answer in time"))??;
    socket
        .send(Message::Text(format!("type:button\nname:{name}\n\n")))
        .await?;
    // Let the press reach the TV before the socket drops.
    tokio::time::sleep(Duration::from_millis(120)).await;
    let _ = socket.close(None).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_register_payload_carries_the_key_only_when_held() {
        let bare = register_payload(None);
        assert_eq!(bare["pairingType"], "PROMPT");
        assert!(bare.get("client-key").is_none());
        assert!(bare["manifest"]["permissions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p == "CONTROL_AUDIO"));
        let keyed = register_payload(Some("abc123"));
        assert_eq!(keyed["client-key"], "abc123");
    }
}
