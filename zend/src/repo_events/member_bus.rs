//! The bus over this member's sockets: each gateway has its own, so a publish
//! goes to every gateway the member holds a socket to, and a request is asked
//! of each until one has a subscriber (`docs/mcp_file_tools.md` §5.2,
//! [`host::cluster::bus`]). The streams go to each gateway's bus group, which
//! is the member's alone, so no key is presented.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use axum::http::Method;
use cluster::member::AppMember;
use host::cluster::bus::{self, SEND_WAIT};
use host::cluster::up::send_over;
use serde_json::{json, Value};
use tokio::runtime::Handle;

use super::{Bus, Unasked};

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
                    Some(SEND_WAIT),
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
        let (tx, rx) = mpsc::channel();
        let (member, topic, reply) = (self.member.clone(), topic.to_string(), reply.to_string());
        self.runtime.spawn(async move {
            let _ = tx.send(bus::ask(&member, &topic, payload, &reply, wait).await);
        });
        rx.recv()
            .map_err(|_| Unasked::Failed("the request was dropped".into()))?
    }
}
