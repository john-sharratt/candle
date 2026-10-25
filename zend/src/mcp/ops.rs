//! What the operations tools share (`docs/claude_zend_gateway.md` §8, §11): who
//! may run them, how a request reaches the cluster, and the gate in front of
//! every decision.
//!
//! A request goes up this member's own socket to a gateway
//! ([`host::cluster::up::send_up`]), stating the caller's identity, so the
//! gateway routes it as it routes the caller's own browser — to every member
//! for `/v1/status`, to one named by `x-tokera-node` for a command — and the
//! member that runs a command checks the caller's role and records them in the
//! coordination log. Nothing here holds a key of its own.

use std::sync::Arc;
use std::time::Duration;

use axum::http::{HeaderName, HeaderValue, Method};
use cluster::gateway::KEEP;
use cluster::member::AppMember;
use host::cluster::up::{send_to_group, send_up, Reply};
use serde_json::Value;
use web::cluster::keeps::HOST_HEADER as KEEP_HOST_HEADER;

use super::caller::Caller;
use crate::session::ZendSession;

/// What a call carries as `confirm` once the user has approved a decision —
/// the endpoint's own gate (§8.2), in front of the members' checks.
pub const APPROVED: &str = "approved by the user";

/// The header naming the member a request is for.
pub const NODE_HEADER: &str = "x-tokera-node";

/// How long a request up waits for its answer when it asks for no longer.
pub const ANSWER_WITHIN: Duration = Duration::from_secs(30);

/// Refuse a caller who is not an Admin, naming the tool.
pub fn admin_only(caller: &Caller, tool: &str) -> Result<(), String> {
    if caller.is_admin() {
        Ok(())
    } else {
        Err(format!(
            "`{tool}` is an Admin's: it reaches the cluster's members, agents and keep, and you \
             are `{}`",
            caller.role
        ))
    }
}

/// Refuse a decision the user has not approved.
pub fn approved(confirm: Option<&str>, what: &str) -> Result<(), String> {
    if confirm == Some(APPROVED) {
        Ok(())
    } else {
        Err(format!(
            "{what} is a decision: it changes what the cluster runs or who is in it. Ask the user, \
             and only with their approval call again with `confirm: \"{APPROVED}\"`"
        ))
    }
}

/// `text` as a member's node: sixteen hex digits, lowercase.
pub fn node(text: &str) -> Result<String, String> {
    let text = text.trim();
    if text.len() == 16 && text.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(text.to_ascii_lowercase())
    } else {
        Err(format!(
            "`{text}` is not a member: a member is its 16-hex-digit node, as `cluster_status` \
             lists it"
        ))
    }
}

/// This member's way up to the cluster.
pub fn member(session: &ZendSession) -> Result<Arc<AppMember>, String> {
    session.cluster_member().ok_or_else(|| {
        "this zend member is not in a cluster, so it has no way to the others: run the \
         operation on a clustered member"
            .into()
    })
}

/// Send `method path` up for `caller`, to member `node` when named, with
/// `body`, reading the answer no longer than `read_for`.
pub async fn up(
    session: &ZendSession,
    caller: &Caller,
    method: Method,
    path: &str,
    node: Option<&str>,
    body: Option<&Value>,
    read_for: Duration,
) -> Result<Reply, String> {
    let member = member(session)?;
    let mut headers = caller.headers();
    if let Some(n) = node {
        headers.push((
            HeaderName::from_static(NODE_HEADER),
            HeaderValue::from_str(n).map_err(|e| e.to_string())?,
        ));
    }
    let bytes = body.map(|b| b.to_string().into_bytes()).unwrap_or_default();
    send_up(&member, method, path, &headers, bytes, Some(read_for))
        .await
        .map_err(|e| format!("the gateway could not be reached: {e:#}"))
}

/// Send `method path` for `caller` to the keep of machine `host`, with `body`,
/// reading the answer no longer than `read_for`.
pub async fn to_keep(
    session: &ZendSession,
    caller: &Caller,
    method: Method,
    path: &str,
    host: &str,
    body: Option<&Value>,
    read_for: Duration,
) -> Result<Reply, String> {
    let member = member(session)?;
    let mut headers = caller.headers();
    headers.push((
        HeaderName::from_static(KEEP_HOST_HEADER),
        HeaderValue::from_str(host)
            .map_err(|e| format!("`{host}` is not a machine's name: {e}"))?,
    ));
    let bytes = body.map(|b| b.to_string().into_bytes()).unwrap_or_default();
    send_to_group(&member, KEEP, method, path, &headers, bytes, Some(read_for))
        .await
        .map_err(|e| format!("the gateway could not be reached: {e:#}"))
}

/// What an answer's body says went wrong: its `detail`, its `error`, or the
/// text itself.
pub fn why(reply: &Reply) -> String {
    let text = String::from_utf8_lossy(&reply.body);
    match serde_json::from_str::<Value>(&text) {
        Ok(v) => v
            .get("detail")
            .or_else(|| v.get("error"))
            .and_then(Value::as_str)
            .map_or_else(|| text.trim().to_string(), str::to_string),
        Err(_) => text.trim().to_string(),
    }
}

/// An answer's body as JSON, or why it is not.
pub fn json_of(reply: &Reply) -> Result<Value, String> {
    if !(200..300).contains(&reply.status) {
        return Err(format!("{} — {}", reply.status, why(reply)));
    }
    serde_json::from_slice(&reply.body)
        .map_err(|e| format!("the answer was not JSON ({e}): {}", why(reply)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Bytes;
    use web::auth::Role;

    use crate::mcp::caller::Person;

    fn caller(role: Role) -> Caller {
        Caller {
            person: Person {
                provider: "google".into(),
                sub: "u".into(),
                email: String::new(),
                name: String::new(),
            },
            role,
        }
    }

    #[test]
    fn only_an_admin_runs_operations() {
        assert!(admin_only(&caller(Role::Admin), "cluster_command").is_ok());
        assert_eq!(
            admin_only(&caller(Role::User), "cluster_command").unwrap_err(),
            "`cluster_command` is an Admin's: it reaches the cluster's members, agents and keep, \
             and you are `user`"
        );
    }

    /// **A decision needs the user's approval, in exactly those words.**
    #[test]
    fn a_decision_is_refused_without_the_users_approval() {
        assert!(approved(Some(APPROVED), "rollout").is_ok());
        for confirm in [None, Some("yes"), Some("approved")] {
            let err = approved(confirm, "rollout").unwrap_err();
            assert!(err.starts_with("rollout is a decision"), "{err}");
            assert!(err.contains("confirm: \"approved by the user\""), "{err}");
        }
    }

    #[test]
    fn a_member_is_sixteen_hex_digits() {
        assert_eq!(node("31F662EE0EDBAE23").unwrap(), "31f662ee0edbae23");
        for bad in ["31f6", "31f662ee0edbae2z", ""] {
            assert!(node(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_refusal_says_what_the_body_said() {
        let reply = |status, body: &str| Reply {
            status,
            body: Bytes::from(body.to_string()),
            complete: true,
        };
        assert_eq!(
            why(&reply(409, r#"{"error":"busy","detail":"a job runs"}"#)),
            "a job runs"
        );
        assert_eq!(why(&reply(500, "plain")), "plain");
        assert_eq!(
            json_of(&reply(403, r#"{"error":"forbidden"}"#)).unwrap_err(),
            "403 — forbidden"
        );
        assert_eq!(json_of(&reply(200, r#"{"a":1}"#)).unwrap()["a"], 1);
    }
}
