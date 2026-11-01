//! CASTv2, the Google Cast control channel: a `CastMessage` protobuf framed by
//! a four-byte big-endian length over TLS on port 8009. The device presents a
//! self-signed certificate, so the connector accepts it.
//!
//! The `CastMessage` has only a handful of fields, so it is hand-encoded here
//! rather than pulling in a protobuf toolchain. Its payload is a JSON string in
//! a namespace — `tp.connection` to open a virtual connection, `receiver` for
//! power/volume/launch, `media` for playback — and a request carries a
//! `requestId` the matching reply echoes.

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use native_tls::TlsConnector as NativeConnector;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_native_tls::{TlsConnector, TlsStream};

/// The control channel's TLS port.
pub const PORT: u16 = 8009;

pub const NS_CONNECTION: &str = "urn:x-cast:com.google.cast.tp.connection";
pub const NS_HEARTBEAT: &str = "urn:x-cast:com.google.cast.tp.heartbeat";
pub const NS_RECEIVER: &str = "urn:x-cast:com.google.cast.receiver";
pub const NS_MEDIA: &str = "urn:x-cast:com.google.cast.media";

/// The sender this library speaks as, and the device's root receiver.
pub const SOURCE: &str = "sender-0";
pub const RECEIVER: &str = "receiver-0";

/// Append a protobuf varint.
fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            break;
        }
    }
}

/// Read a protobuf varint at `*i`, advancing it.
fn read_varint(buf: &[u8], i: &mut usize) -> Option<u64> {
    let mut shift = 0;
    let mut v = 0u64;
    loop {
        let byte = *buf.get(*i)?;
        *i += 1;
        v |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(v);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

fn field_varint(out: &mut Vec<u8>, field: u8, v: u64) {
    out.push(field << 3); // wire type 0 (varint)
    put_varint(out, v);
}

fn field_string(out: &mut Vec<u8>, field: u8, s: &str) {
    out.push((field << 3) | 2); // wire type 2 (length-delimited)
    put_varint(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

/// Encode a `CastMessage`: protocol version 0, a string source/destination/
/// namespace, payload type STRING, and the UTF-8 payload.
pub fn encode(source: &str, dest: &str, namespace: &str, payload: &str) -> Vec<u8> {
    let mut m = Vec::new();
    field_varint(&mut m, 1, 0); // protocol_version = CASTV2_1_0
    field_string(&mut m, 2, source);
    field_string(&mut m, 3, dest);
    field_string(&mut m, 4, namespace);
    field_varint(&mut m, 5, 0); // payload_type = STRING
    field_string(&mut m, 6, payload);
    m
}

/// A decoded `CastMessage` — only the namespace and the JSON payload, which is
/// all a reply is read for.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Msg {
    pub namespace: String,
    pub payload: String,
}

/// Decode a `CastMessage`, keeping its namespace (field 4) and UTF-8 payload
/// (field 6) and skipping the rest.
pub fn decode(buf: &[u8]) -> Option<Msg> {
    let mut i = 0;
    let mut msg = Msg::default();
    while i < buf.len() {
        let tag = buf[i];
        i += 1;
        let field = tag >> 3;
        match tag & 7 {
            0 => {
                read_varint(buf, &mut i)?;
            }
            2 => {
                let len = read_varint(buf, &mut i)? as usize;
                let end = i.checked_add(len)?;
                let bytes = buf.get(i..end)?;
                i = end;
                match field {
                    4 => msg.namespace = String::from_utf8_lossy(bytes).into_owned(),
                    6 => msg.payload = String::from_utf8_lossy(bytes).into_owned(),
                    _ => {}
                }
            }
            _ => return None,
        }
    }
    Some(msg)
}

/// An open control channel to one Cast device.
pub struct Conn {
    stream: TlsStream<TcpStream>,
    answer: Duration,
}

impl Conn {
    /// Open the TLS control channel to `address`, accepting its self-signed
    /// certificate.
    pub async fn open(address: Ipv4Addr, answer: Duration) -> Result<Conn> {
        let native = NativeConnector::builder()
            .danger_accept_invalid_certs(true)
            .danger_accept_invalid_hostnames(true)
            .build()?;
        let connector = TlsConnector::from(native);
        let tcp = timeout(answer, TcpStream::connect((address, PORT)))
            .await
            .map_err(|_| anyhow!("the device did not answer in time"))??;
        let stream = timeout(answer, connector.connect(&address.to_string(), tcp))
            .await
            .map_err(|_| anyhow!("the TLS handshake did not finish in time"))??;
        Ok(Conn { stream, answer })
    }

    async fn write(&mut self, dest: &str, namespace: &str, payload: &str) -> Result<()> {
        let body = encode(SOURCE, dest, namespace, payload);
        let mut frame = (body.len() as u32).to_be_bytes().to_vec();
        frame.extend_from_slice(&body);
        self.stream.write_all(&frame).await?;
        self.stream.flush().await?;
        Ok(())
    }

    async fn read(&mut self) -> Result<Msg> {
        let read = async {
            let mut len = [0u8; 4];
            self.stream.read_exact(&mut len).await?;
            let n = u32::from_be_bytes(len) as usize;
            let mut body = vec![0u8; n];
            self.stream.read_exact(&mut body).await?;
            Ok::<Vec<u8>, anyhow::Error>(body)
        };
        let body = timeout(self.answer, read)
            .await
            .map_err(|_| anyhow!("the device went quiet"))??;
        decode(&body).ok_or_else(|| anyhow!("the device sent a frame that did not parse"))
    }

    /// Open a virtual connection to `dest` — required before any command to it,
    /// whether the root receiver or a running app's transport.
    pub async fn connect_to(&mut self, dest: &str) -> Result<()> {
        self.write(
            dest,
            NS_CONNECTION,
            r#"{"type":"CONNECT","userAgent":"effector-bridge"}"#,
        )
        .await
    }

    /// Close the virtual connection to `dest` — a clean teardown the device
    /// expects before the socket drops.
    pub async fn disconnect(&mut self, dest: &str) -> Result<()> {
        self.write(dest, NS_CONNECTION, r#"{"type":"CLOSE"}"#).await
    }

    /// Send a request to `dest` on `namespace` and return the first reply on
    /// that namespace whose `requestId` matches (or a status that omits one).
    /// `payload` must already carry the `requestId`.
    pub async fn request(
        &mut self,
        dest: &str,
        namespace: &str,
        payload: &str,
        request_id: u64,
    ) -> Result<serde_json::Value> {
        self.write(dest, namespace, payload).await?;
        loop {
            let msg = self.read().await?;
            if msg.namespace == NS_HEARTBEAT {
                // Keep the channel alive through a multi-step flow: a receiver
                // that pings and is not ponged drops the connection.
                if msg.payload.contains("PING") {
                    let _ = self
                        .write(RECEIVER, NS_HEARTBEAT, r#"{"type":"PONG"}"#)
                        .await;
                }
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg.payload) else {
                continue;
            };
            if msg.namespace != namespace {
                continue;
            }
            match v.get("requestId").and_then(serde_json::Value::as_u64) {
                Some(id) if id == request_id => return Ok(v),
                Some(_) => continue,
                None => return Ok(v),
            }
        }
    }

    /// Send a command to `dest` on `namespace` without awaiting a reply.
    pub async fn tell(&mut self, dest: &str, namespace: &str, payload: &str) -> Result<()> {
        self.write(dest, namespace, payload).await
    }
}

/// Fail if a Cast reply carried an error type rather than a status.
pub fn not_an_error(v: &serde_json::Value) -> Result<()> {
    let reason = v
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .map(|r| format!(" ({r})"))
        .unwrap_or_default();
    match v.get("type").and_then(serde_json::Value::as_str) {
        Some("INVALID_REQUEST") => bail!("the device rejected the request{reason}"),
        Some("INVALID_PLAYER_STATE") => bail!("the player was not in a state for that{reason}"),
        Some("LAUNCH_ERROR") => bail!("the device could not launch it{reason}"),
        Some("LOAD_FAILED") => bail!("the device could not load it{reason}"),
        Some("LOAD_CANCELLED") => bail!("the load was cancelled{reason}"),
        Some("ERROR") => {
            let code = v
                .get("detailedErrorCode")
                .and_then(serde_json::Value::as_i64)
                .map(|c| format!(" (code {c})"))
                .unwrap_or_default();
            bail!("the device reported an error{reason}{code}")
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_round_trip() {
        for v in [0u64, 1, 127, 128, 300, 16384, 1_000_000] {
            let mut out = Vec::new();
            put_varint(&mut out, v);
            let mut i = 0;
            assert_eq!(read_varint(&out, &mut i), Some(v));
            assert_eq!(i, out.len());
        }
    }

    /// **A message is the CastMessage the device expects, field by field.**
    #[test]
    fn a_message_is_encoded_field_by_field() {
        assert_eq!(
            encode("a", "b", "c", "d"),
            vec![
                0x08, 0x00, 0x12, 0x01, 0x61, 0x1A, 0x01, 0x62, 0x22, 0x01, 0x63, 0x28, 0x00, 0x32,
                0x01, 0x64
            ]
        );
    }

    /// **A message decodes back to its namespace and payload**, including a
    /// payload long enough to need a two-byte length varint.
    #[test]
    fn a_message_round_trips() {
        let payload = format!("{{\"type\":\"LOAD\",\"pad\":\"{}\"}}", "x".repeat(200));
        let bytes = encode(SOURCE, RECEIVER, NS_MEDIA, &payload);
        let got = decode(&bytes).unwrap();
        assert_eq!(got.namespace, NS_MEDIA);
        assert_eq!(got.payload, payload);
    }

    #[test]
    fn an_error_reply_is_an_error() {
        assert!(not_an_error(&serde_json::json!({"type": "LOAD_FAILED"})).is_err());
        assert!(not_an_error(&serde_json::json!({"type": "INVALID_PLAYER_STATE"})).is_err());
        let launch =
            not_an_error(&serde_json::json!({"type": "LAUNCH_ERROR", "reason": "NOT_FOUND"}));
        assert!(launch.unwrap_err().to_string().contains("NOT_FOUND"));
        let coded = not_an_error(&serde_json::json!({"type": "ERROR", "detailedErrorCode": 905}));
        assert!(coded.unwrap_err().to_string().contains("905"));
        assert!(not_an_error(&serde_json::json!({"type": "MEDIA_STATUS"})).is_ok());
    }
}
