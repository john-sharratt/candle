//! `Aggregate(Lines)` (§5.2): one browser WebSocket, every ready member's
//! lines interleaved, each tagged with the member it came from. `/ws/logs`
//! sends one JSON object per message; the gateway adds `node`, `host` and
//! `member` — the two as a reader shows them, `whiteworld 31f662ee` — to each,
//! and wraps anything that is not an object as `{line, node, host, member}`.
//! Every member is read the same way, the one a request came up from too.
//!
//! A member whose socket drops stops contributing; the browser's socket stays
//! up while any member's does, and closes when the last one ends.

use std::sync::Arc;

use axum::extract::ws::{Message as BrowserMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{FromRequestParts, Request};
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use futures::{SinkExt, StreamExt};
use serde::de::IgnoredAny;
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::client_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message as MemberMessage;

use super::route::Context;
use super::Gateway;
use crate::proxy::name_caller;

/// Lines waiting for the browser before a member's reader waits.
const QUEUE: usize = 1024;

/// How a reader names the member at `node` on `host`: the machine and the
/// node's first eight hex digits, `whiteworld 31f662ee`.
pub fn member_name(node: u64, host: &str) -> String {
    let short = &format!("{node:016x}")[..8];
    if host.is_empty() {
        short.to_string()
    } else {
        format!("{host} {short}")
    }
}

/// How one member's lines are tagged: the three fields, written once.
pub struct Tag {
    node: String,
    host: String,
    member: String,
    /// `,"node":…,"host":…,"member":…}` — spliced before an object's last
    /// brace.
    suffix: String,
}

impl Tag {
    /// The tag of the member at `node` on `host`.
    pub fn of(node: u64, host: &str) -> Tag {
        let member = member_name(node, host);
        let node = format!("{node:016x}");
        let text = |s: &str| Value::String(s.to_string()).to_string();
        let suffix = format!(
            ",\"node\":{},\"host\":{},\"member\":{}}}",
            text(&node),
            text(host),
            text(&member)
        );
        Tag {
            node,
            host: host.to_string(),
            member,
            suffix,
        }
    }

    /// `line`, tagged. A line that is a JSON object keeps its own bytes, the
    /// fields spliced in before its closing brace — one already carrying one
    /// of them then has it twice, and a reader keeps the last. Anything else
    /// is wrapped as `{line, node, host, member}`.
    pub fn tag(&self, line: &str) -> String {
        let object = line.trim_end();
        let is_object = object.starts_with('{')
            && object.ends_with('}')
            && serde_json::from_str::<IgnoredAny>(line).is_ok();
        if is_object {
            let open = &object[..object.len() - 1];
            return if open[1..].trim().is_empty() {
                format!("{open}{}", &self.suffix[1..])
            } else {
                format!("{open}{}", self.suffix)
            };
        }
        let mut o = Map::new();
        o.insert("line".into(), Value::String(line.to_string()));
        o.insert("node".into(), Value::String(self.node.clone()));
        o.insert("host".into(), Value::String(self.host.clone()));
        o.insert("member".into(), Value::String(self.member.clone()));
        Value::Object(o).to_string()
    }
}

pub async fn merge(
    gateway: &Arc<Gateway>,
    app: &str,
    nodes: Vec<u64>,
    ctx: &Context<'_>,
    req: Request,
) -> Response {
    let (mut parts, _) = req.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map_or("/".to_string(), |p| p.as_str().to_string());
    let mut headers = HeaderMap::new();
    if let Some(host) = parts.headers.get(header::HOST) {
        headers.insert(header::HOST, host.clone());
    }
    if let Some(id) = ctx.identity {
        name_caller(&mut headers, id);
    }
    let ws = match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
        Ok(ws) => ws,
        Err(rejection) => return rejection.into_response(),
    };
    gateway.hub.traffic.aggregate(0);
    let gateway = gateway.clone();
    let app = app.to_string();
    ws.on_upgrade(move |browser| relay(gateway, app, nodes, path, headers, browser))
}

async fn relay(
    gateway: Arc<Gateway>,
    app: String,
    nodes: Vec<u64>,
    path: String,
    headers: HeaderMap,
    browser: WebSocket,
) {
    let _open = gateway.hub.traffic.stream();
    let (tx, mut lines) = mpsc::channel::<String>(QUEUE);
    for node in nodes {
        let tx = tx.clone();
        let gateway = gateway.clone();
        let app = app.clone();
        let path = path.clone();
        let headers = headers.clone();
        tokio::spawn(async move {
            if let Err(e) = read_member(&gateway, &app, node, &path, headers, tx).await {
                tracing::debug!(node = %format!("{node:016x}"), error = %e, "merged stream ended");
            }
        });
    }
    drop(tx);
    let (mut sink, mut stream) = browser.split();
    loop {
        tokio::select! {
            line = lines.recv() => match line {
                Some(text) => {
                    if sink.send(BrowserMessage::Text(text)).await.is_err() {
                        return;
                    }
                }
                None => {
                    let _ = sink.close().await;
                    return;
                }
            },
            incoming = stream.next() => match incoming {
                Some(Ok(BrowserMessage::Close(_))) | Some(Err(_)) | None => return,
                Some(Ok(_)) => {}
            },
        }
    }
}

async fn read_member(
    gateway: &Gateway,
    app: &str,
    node: u64,
    path: &str,
    headers: HeaderMap,
    tx: mpsc::Sender<String>,
) -> Result<(), String> {
    let conn = gateway
        .hub
        .open(app, node, app)
        .ok_or("the member's socket is gone")?;
    let tag = Tag::of(
        node,
        &gateway.hub.member_host(app, node).unwrap_or_default(),
    );
    let mut request = format!("ws://{app}{path}")
        .into_client_request()
        .map_err(|e| e.to_string())?;
    for (name, value) in &headers {
        request.headers_mut().insert(name.clone(), value.clone());
    }
    let (mut ws, _) = client_async(request, conn)
        .await
        .map_err(|e| e.to_string())?;
    while let Some(message) = ws.next().await {
        match message.map_err(|e| e.to_string())? {
            MemberMessage::Text(text) => {
                if tx.send(tag.tag(&text)).await.is_err() {
                    return Ok(());
                }
            }
            MemberMessage::Close(_) => return Ok(()),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    fn tag(line: &str, node: u64, host: &str) -> String {
        Tag::of(node, host).tag(line)
    }

    #[test]
    fn a_line_is_tagged_with_its_member() {
        let tagged: Value = serde_json::from_str(&tag(
            r#"{"msg":"hi","seq":3}"#,
            0x31f6_62ee_0edb_ae23,
            "whiteworld",
        ))
        .unwrap();
        assert_eq!(
            tagged,
            json!({
                "msg": "hi",
                "seq": 3,
                "node": "31f662ee0edbae23",
                "host": "whiteworld",
                "member": "whiteworld 31f662ee",
            })
        );
        let wrapped: Value = serde_json::from_str(&tag("plain", 1, ".6")).unwrap();
        assert_eq!(
            wrapped,
            json!({
                "node": "0000000000000001",
                "host": ".6",
                "member": ".6 00000000",
                "line": "plain",
            })
        );
    }

    /// **An object keeps its own bytes**, the member's fields spliced before
    /// its closing brace — an empty one without a leading comma, whitespace
    /// after the brace dropped; a line braced but not JSON, and one that is
    /// not an object, are wrapped.
    #[test]
    fn an_object_line_is_spliced_exactly() {
        let t = Tag::of(1, "h\"q");
        let fields = r#""node":"0000000000000001","host":"h\"q","member":"h\"q 00000000""#;
        assert_eq!(t.tag(r#"{"a": 1}"#), format!(r#"{{"a": 1,{fields}}}"#));
        assert_eq!(t.tag("{}"), format!("{{{fields}}}"));
        assert_eq!(t.tag("{ }"), format!("{{ {fields}}}"));
        assert_eq!(t.tag("{\"a\":1}  \n"), format!(r#"{{"a":1,{fields}}}"#));
        let wrapped = |line: &str| -> Value { serde_json::from_str(&t.tag(line)).unwrap() };
        let want = |line: &str| {
            json!({
                "line": line,
                "node": "0000000000000001",
                "host": "h\"q",
                "member": "h\"q 00000000",
            })
        };
        assert_eq!(wrapped("{oops}"), want("{oops}"));
        assert_eq!(wrapped("[1, 2]"), want("[1, 2]"));
        assert_eq!(wrapped("plain"), want("plain"));
    }

    #[test]
    fn a_member_with_no_host_is_its_node() {
        assert_eq!(member_name(0xabcd_ef01_2345_6789, ""), "abcdef01");
    }
}
