//! `log_read {source, level?, since?, limit?}`: one source's newest lines,
//! oldest first.

use serde_json::{json, Value};

use super::{cluster_text, search_properties, site_entry, Source};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::cluster_logs::render;
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "log_read";

pub fn definition() -> Value {
    let mut properties = search_properties();
    properties["source"] = json!({"type": "string", "description": "`cluster`, a site such as `yenbot`, or one of its sources such as `yenbot/recipes` — as `log_list` names them."});
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read one log's newest lines, oldest first: `cluster` for the machines running the \
            user's daemons, `yenbot` for Yen Bots itself, `yenbot/<app>` for one app's browser errors, \
            workflow runs and check refusals. Narrow with `level` and `since` (UTC). To find something, \
            use `log_grep`. Cite the lines you rely on.",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["source"],
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let source = match string(args, "source").and_then(Source::parse) {
        Ok(s) => s,
        Err(e) => return Answer::error(e),
    };
    match source {
        Source::Cluster => cluster_text(call, args).await,
        Source::Site { site, sub } => {
            let entry = site_entry(call, &site, sub.as_deref(), args).await;
            Answer::ok(render(&json!({"members": [entry]})))
        }
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}
