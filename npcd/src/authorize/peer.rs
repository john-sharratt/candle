//! Asking another member for an authorization while this one draws
//! (`docs/guest_draw_orchestration.md` §2): one seam, [`PeerAuthorizer`], whose
//! implementation sends `POST /v1/prompt/authorize` up this member's own
//! cluster socket. The gateway places it on another member when one can take
//! it (§3), so this member chooses no peer.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{HeaderName, HeaderValue, Method};
use cluster::member::PREFER_PEER;
use host::cluster::up::send_up;
use serde_json::{json, Value};

use super::token::Authorization;
use crate::engine::runtime::Runtime;

/// The route a peer authorizes on.
pub const PATH: &str = "/v1/prompt/authorize";

/// How long a peer's check is waited for before it counts as failed.
pub const WAIT: Duration = Duration::from_secs(30);

/// What a peer answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerVerdict {
    /// Cleared: the authorization it signed, still to be verified here.
    Allowed(Authorization),
    /// Declined, with the reason it gave.
    Refused(String),
    /// No verdict: the peer, the gateway or the link failed, or it ran out of
    /// time.
    Failed(String),
}

/// Another member that checks prompts.
pub trait PeerAuthorizer {
    /// Whether another member could be asked.
    fn available(&self) -> bool;

    /// Ask it to authorize `prompt` for `purpose`.
    fn authorize(&self, purpose: &str, prompt: &str) -> impl Future<Output = PeerVerdict> + Send;
}

/// The peers in this member's cluster.
pub struct ClusterPeers {
    rt: Arc<Runtime>,
}

impl ClusterPeers {
    pub fn new(rt: Arc<Runtime>) -> Self {
        Self { rt }
    }
}

impl PeerAuthorizer for ClusterPeers {
    /// Another npcd member ready — a member running no characters included: it
    /// still runs the model the check asks.
    fn available(&self) -> bool {
        let me = self.rt.standing.node();
        let (ready, _) = self.rt.npcd_members();
        ready.iter().any(|m| Some(m.node) != me)
    }

    fn authorize(&self, purpose: &str, prompt: &str) -> impl Future<Output = PeerVerdict> + Send {
        let rt = Arc::clone(&self.rt);
        let body = json!({ "purpose": purpose, "prompt": prompt }).to_string();
        async move {
            let Some(member) = rt.standing.member() else {
                return PeerVerdict::Failed("no cluster is joined".into());
            };
            // Asked of a peer: this member's engine is busy drawing.
            let peer = [(
                HeaderName::from_static(PREFER_PEER),
                HeaderValue::from_static("1"),
            )];
            match send_up(
                member,
                Method::POST,
                PATH,
                &peer,
                body.into_bytes(),
                Some(WAIT),
            )
            .await
            {
                Ok(reply) if !reply.complete => {
                    PeerVerdict::Failed(format!("no verdict within {}s", WAIT.as_secs()))
                }
                Ok(reply) => verdict(reply.status, &reply.body),
                Err(e) => PeerVerdict::Failed(format!("{e:#}")),
            }
        }
    }
}

/// What a peer's answer says. Only a body whose `error` is `prompt_declined`
/// is a refusal: a gateway's own 403, 404 or 503 is a failure to ask, not an
/// answer.
pub fn verdict(status: u16, body: &[u8]) -> PeerVerdict {
    let v: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    if status == 200 {
        return match serde_json::from_value::<Authorization>(v["authorization"].clone()) {
            Ok(a) => PeerVerdict::Allowed(a),
            Err(e) => PeerVerdict::Failed(format!("the peer's authorization is unreadable: {e}")),
        };
    }
    if v["error"] == "prompt_declined" {
        let why = v["detail"].as_str().unwrap_or("declined").to_string();
        return PeerVerdict::Refused(why);
    }
    PeerVerdict::Failed(format!(
        "the peer answered {status}: {}",
        String::from_utf8_lossy(body)
            .chars()
            .take(200)
            .collect::<String>()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorize::token::{Signer, IMAGE};

    /// **Only the peer's own `prompt_declined` is a refusal**; a 200 carries
    /// the authorization; anything else — a gateway's forbidden, a member not
    /// there, a check that could not run — is a failure to ask.
    #[test]
    fn a_peers_answer_is_read_by_its_body() {
        let a = Signer::new([3; 32]).sign(IMAGE, "x", 0);
        let ok = json!({ "authorization": a }).to_string();
        assert_eq!(verdict(200, ok.as_bytes()), PeerVerdict::Allowed(a));
        let no = json!({ "error": "prompt_declined", "detail": "not this" }).to_string();
        assert_eq!(
            verdict(403, no.as_bytes()),
            PeerVerdict::Refused("not this".into())
        );
        let gw = json!({ "error": "forbidden", "detail": "admins only" }).to_string();
        assert!(matches!(
            verdict(403, gw.as_bytes()),
            PeerVerdict::Failed(_)
        ));
        let down = json!({ "error": "check_unavailable", "retry": true }).to_string();
        assert!(matches!(
            verdict(503, down.as_bytes()),
            PeerVerdict::Failed(_)
        ));
        assert!(matches!(verdict(200, b"{}"), PeerVerdict::Failed(_)));
        assert!(matches!(
            verdict(502, b"bad gateway"),
            PeerVerdict::Failed(_)
        ));
    }
}
