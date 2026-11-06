//! The bus over this member's sockets: each gateway has its own, so a publish
//! goes to every gateway the member holds a socket to, and a request is asked
//! of each until one has a subscriber (`docs/mcp_file_tools.md` §5.2). The
//! streams go to each gateway's bus group, which is the member's alone, so no
//! key is presented.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use axum::http::Method;
use cluster::member::AppMember;
use host::cluster::up::{send_over, Reply};
use serde_json::{json, Value};
use tokio::runtime::Handle;

use super::{Bus, Unasked};

/// How long a publish waits on one gateway before giving up on it.
const PUBLISH_WAIT: Duration = Duration::from_secs(5);

pub struct MemberBus {
    member: Arc<AppMember>,
    runtime: Handle,
}

impl MemberBus {
    pub fn new(member: Arc<AppMember>, runtime: Handle) -> MemberBus {
        MemberBus { member, runtime }
    }
}

impl Bus for MemberBus {
    fn publish(&self, topic: &str, payload: Value) {
        let body = json!({"topic": topic, "payload": payload}).to_string();
        let topic = topic.to_string();
        for conn in self.member.open_bus_all() {
            let body = body.clone().into_bytes();
            let topic = topic.clone();
            // Not waited for: the push it announces has landed, and a gateway
            // that missed the message is read again by the subscriber's own
            // resynchronisation.
            self.runtime.spawn(async move {
                let sent = send_over(
                    conn,
                    Method::POST,
                    "/gateway/events/publish",
                    &[],
                    body,
                    Some(PUBLISH_WAIT),
                )
                .await;
                match sent {
                    Ok(r) if r.status == 200 => {}
                    Ok(r) => {
                        tracing::warn!(topic, status = r.status, "bus: a gateway refused a publish")
                    }
                    Err(e) => tracing::warn!(topic, "bus: a publish did not reach a gateway: {e}"),
                }
            });
        }
    }

    fn ask(
        &self,
        topic: &str,
        payload: Value,
        reply: &str,
        wait: Duration,
    ) -> Result<Value, Unasked> {
        let conns = self.member.open_bus_all();
        if conns.is_empty() {
            return Err(Unasked::Failed("no socket to a gateway is open".into()));
        }
        let body = json!({
            "topic": topic,
            "payload": payload,
            "reply": reply,
            "timeout_ms": wait.as_millis() as u64,
        })
        .to_string();
        let (tx, rx) = mpsc::channel();
        self.runtime.spawn(async move {
            let mut failed: Option<String> = None;
            for conn in conns {
                let sent = send_over(
                    conn,
                    Method::POST,
                    "/gateway/events/request",
                    &[],
                    body.clone().into_bytes(),
                    Some(wait + PUBLISH_WAIT),
                )
                .await;
                match heard(sent.map_err(|e| e.to_string())) {
                    // Another gateway may hold the subscriber — the one this
                    // could not reach among them.
                    Err(Unasked::NoSubscriber) => {}
                    Err(Unasked::Failed(why)) => {
                        failed.get_or_insert(why);
                    }
                    answer => {
                        let _ = tx.send(answer);
                        return;
                    }
                }
            }
            // A gateway that could not be asked may be the one the gate is on.
            let _ = tx.send(Err(failed.map_or(Unasked::NoSubscriber, Unasked::Failed)));
        });
        rx.recv()
            .map_err(|_| Unasked::Failed("the request was dropped".into()))?
    }
}

/// What one gateway said to a request.
fn heard(sent: Result<Reply, String>) -> Result<Value, Unasked> {
    match sent {
        Ok(r) if r.status == 200 => serde_json::from_slice(&r.body)
            .map_err(|e| Unasked::Failed(format!("the answer is not JSON: {e}"))),
        Ok(r) if r.status == 404 => Err(Unasked::NoSubscriber),
        Ok(r) if r.status == 504 => Err(Unasked::Timeout),
        Ok(r) => Err(Unasked::Failed(format!(
            "the gateway answered {}: {}",
            r.status,
            String::from_utf8_lossy(&r.body)
        ))),
        Err(e) => Err(Unasked::Failed(e)),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Bytes;

    use super::*;

    fn reply(status: u16, body: &str) -> Result<Reply, String> {
        Ok(Reply {
            status,
            body: Bytes::from(body.to_string()),
            complete: true,
        })
    }

    /// **Each gateway's answer reads as the bus's**: the gate's payload, no
    /// subscriber, a gate that heard and did not answer, or a failure.
    #[test]
    fn a_gateways_answer_reads_as_the_buss() {
        assert_eq!(heard(reply(200, r#"{"ok":true}"#)), Ok(json!({"ok": true})));
        assert_eq!(heard(reply(404, "")), Err(Unasked::NoSubscriber));
        assert_eq!(heard(reply(504, "")), Err(Unasked::Timeout));
        assert_eq!(
            heard(reply(401, "no")),
            Err(Unasked::Failed("the gateway answered 401: no".into()))
        );
        assert_eq!(
            heard(Err("the gateway did not answer in time".into())),
            Err(Unasked::Failed("the gateway did not answer in time".into()))
        );
        assert!(matches!(heard(reply(200, "{")), Err(Unasked::Failed(_))));
    }
}
