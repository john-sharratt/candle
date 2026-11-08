//! `member_memory {node}`: one member's own `/v1/memory`, whole — where its
//! machine's and its engine's memory went: the host's RAM, the card's VRAM
//! partition, the weights, and the KV arenas by size class
//! (`docs/kv_drain_under_pressure.md` §2a reads the quant mix from it).

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::json_of;
use crate::mcp::ops_tools::OpsCall;

pub const NAME: &str = "member_memory";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "One member's memory, whole, as it reports itself: the host's RAM, the card's \
            VRAM partition (KV regions, the wave tier, expert weights), and the KV arenas by size \
            class — slot size, arenas, reserved and live bytes, how many would pack and free. Read \
            it to see what holds the card and which quant formats the KV is in. `node` is a \
            member's 16-hex-digit node id, as `cluster_status` lists it.",
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
            "/v1/memory",
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
        Err(e) => Answer::error(format!("member {node}'s /v1/memory: {e}")),
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
        assert!(!is_node("WhiteWorld"));
    }
}
