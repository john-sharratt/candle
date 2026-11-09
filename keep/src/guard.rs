//! The guards on stopping a service: a request by hand that would stop the
//! only member serving an app is refused, and so is any switch, restart or
//! stop of a service whose own status refuses it.
//!
//! **A service may refuse for itself.** A status carrying `refuse: "<why>"`
//! says the process is doing something a restart would break. That refusal
//! holds for a trial the cluster asks for too ([`crate::remote`]), since
//! nothing in the cluster knows what the service knows. A process keep does
//! not run refuses the same way for the whole machine (`cluster::reserve`).
//!
//! Every app's quorum is on the service's own `/v1/status`. An app this member
//! is ready for, in a group of two or more, with one member ready, is served by
//! this member alone: a switch now takes it down until something else comes up.
//! On 2026-10-31 a switch by hand landed while the cluster was installing on the
//! other member, and zend was down for three minutes.
//!
//! The cluster's own install asks with [`FORCE_HEADER`], since it has already
//! taken the rollout's lease and a refusal would strand the member; so can a
//! person who means it. A status that cannot be read refuses nothing: a service
//! not answering is not serving anyone.

use cluster::reserve::guard::{reason, status_at};
use serde_json::Value;

pub use cluster::upgrade::switch::FORCE_HEADER;

/// The apps `status`, a service's `/v1/status`, shows this member serving alone.
pub fn served_alone(status: &Value) -> Vec<String> {
    let Some(apps) = status["cluster"]["apps"].as_object() else {
        return Vec::new();
    };
    apps.iter()
        .filter(|(name, app)| {
            let quorum = &app["quorum"];
            status["tenants"][name.as_str()]["state"] == "ready"
                && quorum["max_members"].as_u64().unwrap_or(0) >= 2
                && quorum["ready"].as_u64() == Some(1)
        })
        .map(|(name, _)| name.clone())
        .collect()
}

/// What `status` says when the service refuses to be stopped: its
/// [`reason`], worded with the way past it.
pub fn declined(status: &Value) -> Option<String> {
    reason(status).map(|why| {
        format!(
            "refused: the service says {why}. Wait for it to clear, or send \
             `{FORCE_HEADER}: 1` to do it anyway\n"
        )
    })
}

/// Why the service whose status is at `addr` refuses to be stopped, if it
/// says so itself.
pub fn own_refusal(addr: &str) -> Option<String> {
    declined(&status_at(addr)?)
}

/// Why stopping the service whose status is at `addr` is refused by the
/// last-member guard, if it is.
pub fn refusal(addr: &str) -> Option<String> {
    let alone = served_alone(&status_at(addr)?);
    if alone.is_empty() {
        return None;
    }
    Some(format!(
        "refused: this is the only member serving {}. Wait for another to be ready, or send \
         `{FORCE_HEADER}: 1` to do it anyway\n",
        alone.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn status(state: &str, max_members: u64, ready: u64) -> Value {
        json!({
            "tenants": {"zend": {"state": state}},
            "cluster": {"apps": {"zend": {"quorum": {"max_members": max_members, "ready": ready}}}},
        })
    }

    #[test]
    fn only_a_ready_member_with_no_ready_peer_serves_alone() {
        assert_eq!(served_alone(&status("ready", 2, 1)), ["zend"]);
        assert!(served_alone(&status("ready", 2, 2)).is_empty(), "a peer");
        assert!(
            served_alone(&status("loading", 2, 1)).is_empty(),
            "the one ready is the peer"
        );
        assert!(served_alone(&status("ready", 1, 1)).is_empty(), "no group");
        assert!(served_alone(&json!({})).is_empty(), "no cluster");
    }

    /// **A service's own `refuse` is its refusal**, worded with the way past
    /// it; an empty or missing one, or one that is not text, refuses nothing.
    #[test]
    fn a_service_refuses_for_itself_when_its_status_says_why() {
        assert_eq!(
            declined(&json!({"refuse": "music is playing on deck 1"})).as_deref(),
            Some(
                "refused: the service says music is playing on deck 1. Wait for it to clear, or \
                 send `X-Keep-Force: 1` to do it anyway\n"
            )
        );
        assert_eq!(declined(&json!({"refuse": "  "})), None);
        assert_eq!(declined(&json!({"refuse": null})), None);
        assert_eq!(declined(&json!({"refuse": true})), None);
        assert_eq!(declined(&json!({})), None);
    }

    #[test]
    fn a_status_that_cannot_be_read_refuses_nothing() {
        assert_eq!(refusal("127.0.0.1:1"), None);
        assert_eq!(refusal("not an address"), None);
    }
}
