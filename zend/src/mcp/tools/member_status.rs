//! `member_status {node}`: one member's own `/v1/status`, whole
//! (`docs/zen_tool_reach.md` §4.2) — what `cluster_status` summarises, for
//! what the summary leaves out: each app's replication phase, link,
//! generation, quorum and leases, its catch-ups, its maintenance.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::json_of;
use crate::mcp::ops_tools::OpsCall;

pub const NAME: &str = "member_status";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "One member's own status, whole, as it reports itself: its state and build, each \
            app it hosts with its link to the cluster and its replication — phase (joined, cut off, \
            rejoining), generation, unsent events, catch-ups in flight, quorum and the leases it \
            holds — its upgrade, and each tenant's maintenance. `cluster_status` summarises every \
            member; read this one when the summary does not say enough, such as whether a member \
            is really joined. `node` is a member's 16-hex-digit node id, as `cluster_status` lists it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "minLength": 16, "maxLength": 16, "description": "The member's node id, 16 hex digits."}
            },
            "required": ["node"],
            "additionalProperties": false
        }
    })
}

/// Whether `node` is a member's node id: 16 hex digits.
fn is_node(node: &str) -> bool {
    node.len() == 16 && node.bytes().all(|b| b.is_ascii_hexdigit())
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let node = match string(args, "node") {
        Ok(n) if is_node(n) => n,
        Ok(n) => {
            return Answer::error(format!(
                "`{n}` is not a node id: a member's node is 16 hex digits, as `cluster_status` \
                 lists it"
            ))
        }
        Err(e) => return Answer::error(e),
    };
    let reply = match call
        .up(
            Method::GET,
            "/v1/status",
            Some(node),
            Duration::from_secs(15),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(format!("member {node} did not answer: {e}")),
    };
    match json_of(&reply) {
        Ok(v) => Answer::ok(serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())),
        Err(e) => Answer::error(format!("member {node}'s /v1/status: {e}")),
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_is_sixteen_hex_digits() {
        assert!(is_node("893959beacadc372"));
        assert!(!is_node("893959beacadc37"));
        assert!(!is_node("893959beacadc37g"));
        assert!(!is_node("WhiteWorld"));
    }
}
