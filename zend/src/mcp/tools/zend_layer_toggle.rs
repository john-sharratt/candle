//! `zend_layer_toggle {name}`: flip a projection layer's kill switch
//! (`docs/claude_zend_gateway.md` §8.5) — `POST
//! /v1/substrate/zend/layer/:name/toggle`. The switch is held in each member's
//! memory, so on a clustered member the request goes up and the gateway asks
//! every member; alone, it flips this one. A restart clears it. A Care
//! operation, an Admin's.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of};

pub const NAME: &str = "zend_layer_toggle";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Flip one projection layer out of, or back into, every member's projection — to test \
            whether a layer's content is behind a bad answer. Held in memory: a restart clears it. Answers \
            each member's new state. An Admin's.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The layer."}
            },
            "required": ["name"],
            "additionalProperties": false
        }
    })
}

/// Each member's answer to the toggle, one line each.
pub fn render(name: &str, envelope: &Value) -> String {
    let mut out = format!("`{name}`:");
    for m in envelope
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let host = m.get("host").and_then(Value::as_str).unwrap_or("?");
        let state = match m
            .get("body")
            .and_then(|b| b.get("disabled"))
            .and_then(Value::as_bool)
        {
            Some(true) => "now excluded".to_string(),
            Some(false) => "now projected".to_string(),
            None => format!(
                "no answer ({})",
                m.get("status").map_or_else(|| "?".into(), Value::to_string)
            ),
        };
        out.push_str(&format!("\n- {host}: {state}"));
    }
    out
}

fn single(name: &str, disabled: bool) -> String {
    format!(
        "`{name}` is {} on this member.",
        if disabled {
            "now excluded"
        } else {
            "now projected"
        }
    )
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let name = match string(args, "name") {
        Ok(n) => n,
        Err(e) => return Answer::error(e),
    };
    if ctx.session.cluster_member().is_none() {
        return match ctx.session.toggle_projection_layer(name) {
            Some(disabled) => Answer::ok(single(name, disabled)),
            None => Answer::error(format!(
                "there is no layer `{name}`, or zend is still loading"
            )),
        };
    }
    let reply = ops::up(
        &ctx.session,
        &ctx.caller,
        Method::POST,
        &format!("/v1/substrate/zend/layer/{name}/toggle"),
        None,
        None,
        Duration::from_secs(15),
    )
    .await;
    match reply.and_then(|r| json_of(&r)) {
        Ok(v) => Answer::ok(render(name, &v)),
        Err(e) => Answer::error(format!("`{name}`: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_member_says_where_the_layer_now_stands() {
        let envelope = json!({"members": [
            {"host": "bw", "status": 200, "body": {"name": "code_review", "disabled": true}},
            {"host": "ww", "status": 504}
        ]});
        assert_eq!(
            render("code_review", &envelope),
            "`code_review`:\n- bw: now excluded\n- ww: no answer (504)"
        );
        assert_eq!(
            single("code_review", false),
            "`code_review` is now projected on this member."
        );
    }
}
