//! The last-member guard: a request by hand that would stop the only member
//! serving an app is refused.
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

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::Value;

/// How long the status is waited for.
const TIMEOUT: Duration = Duration::from_secs(3);

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

/// The status at `addr`, `None` when it cannot be read.
fn status_at(addr: &str) -> Option<Value> {
    let addr: SocketAddr = addr.parse().ok()?;
    let read = || -> io::Result<String> {
        let mut stream = TcpStream::connect_timeout(&addr, TIMEOUT)?;
        stream.set_read_timeout(Some(TIMEOUT))?;
        write!(
            stream,
            "GET /v1/status HTTP/1.0\r\nHost: {addr}\r\nx-tokera-email: keep@test.invalid\r\n\r\n"
        )?;
        let mut answer = String::new();
        stream.read_to_string(&mut answer)?;
        Ok(answer)
    };
    let answer = read().ok()?;
    let (_, body) = answer.split_once("\r\n\r\n")?;
    serde_json::from_str(body).ok()
}

/// Why stopping the service whose status is at `addr` is refused, if it is.
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

    #[test]
    fn a_status_that_cannot_be_read_refuses_nothing() {
        assert_eq!(refusal("127.0.0.1:1"), None);
        assert_eq!(refusal("not an address"), None);
    }
}
