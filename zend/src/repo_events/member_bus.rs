//! The bus over this member's sockets: each gateway has its own, so a publish
//! goes to every gateway the member holds a socket to, and a request is asked
//! of each until one has a subscriber (`docs/mcp_file_tools.md` §5.2). The
//! gateway believes a member's stream, so no key is presented.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use axum::http::Method;
use cluster::member::AppMember;
use host::cluster::up::send_over;
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
        for conn in self.member.open_up_all() {
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
        let conns = self.member.open_up_all();
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
            let mut last = Err(Unasked::NoSubscriber);
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
                last = match sent {
                    Ok(r) if r.status == 200 => serde_json::from_slice(&r.body)
                        .map_err(|e| Unasked::Failed(format!("the answer is not JSON: {e}"))),
                    Ok(r) if r.status == 404 => Err(Unasked::NoSubscriber),
                    Ok(r) if r.status == 504 => Err(Unasked::Timeout),
                    Ok(r) => Err(Unasked::Failed(format!(
                        "the gateway answered {}: {}",
                        r.status,
                        String::from_utf8_lossy(&r.body)
                    ))),
                    Err(e) => Err(Unasked::Failed(e.to_string())),
                };
                // Another gateway may hold the subscriber; anything else is
                // this request's answer.
                if last != Err(Unasked::NoSubscriber) {
                    break;
                }
            }
            let _ = tx.send(last);
        });
        rx.recv()
            .map_err(|_| Unasked::Failed("the request was dropped".into()))?
    }
}
