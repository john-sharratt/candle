//! A member's asks on the gateways' bus (`docs/mcp_file_tools.md` §5.2): each
//! gateway has its own bus, so an ask goes to each gateway the member holds a
//! socket to until one has a subscriber — the gate of a repository for zend,
//! the DJ for the house's mode (`docs/house_modes.md` §3). The streams go to
//! each gateway's bus group, which is the member's alone, so no key is
//! presented.

use std::time::Duration;

use axum::http::Method;
use cluster::member::AppMember;
use serde_json::{json, Value};

use super::up::{send_over, Reply};

/// How long a request waits on one gateway past its own answer's wait.
pub const SEND_WAIT: Duration = Duration::from_secs(5);

/// Why an ask on the bus had no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unasked {
    /// No gateway has anything subscribed to the topic.
    NoSubscriber,
    /// Something heard it and did not answer in time.
    Timeout,
    /// No gateway could be reached, or one answered with an error.
    Failed(String),
}

/// Publish `payload` on `topic` and wait at most `wait` for the first
/// payload on `reply`, asking each gateway `member` holds a socket to until
/// one has a subscriber.
pub async fn ask(
    member: &AppMember,
    topic: &str,
    payload: Value,
    reply: &str,
    wait: Duration,
) -> Result<Value, Unasked> {
    let conns = member.open_bus_all();
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
    let mut failed: Option<String> = None;
    for conn in conns {
        let sent = send_over(
            conn,
            Method::POST,
            "/gateway/events/request",
            &[],
            body.clone().into_bytes(),
            Some(wait + SEND_WAIT),
        )
        .await;
        match heard(sent.map_err(|e| e.to_string())) {
            // Another gateway may hold the subscriber.
            Err(Unasked::NoSubscriber) => {}
            // A gateway that could not be asked may be the one it is on.
            Err(Unasked::Failed(why)) => {
                failed.get_or_insert(why);
            }
            answer => return answer,
        }
    }
    Err(failed.map_or(Unasked::NoSubscriber, Unasked::Failed))
}

/// What one gateway said to a request.
pub fn heard(sent: Result<Reply, String>) -> Result<Value, Unasked> {
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

    /// **Each gateway's answer reads as the bus's**: the answerer's payload,
    /// no subscriber, one that heard and did not answer, or a failure.
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
