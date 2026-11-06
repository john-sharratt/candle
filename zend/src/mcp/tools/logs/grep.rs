//! `log_grep {grep, source?, level?, since?, limit?, tenant?, node?}`: a
//! regular expression over one source, or over every source when none is
//! named, each line under its source. `source: "cluster"` is what the
//! worker's `cluster_logs` answers.

use futures::future::join_all;
use serde_json::{json, Value};

use super::{cluster_text, search_properties, site_entry, sites, Source};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::cluster_logs::render;
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "log_grep";

pub fn definition() -> Value {
    let mut properties = search_properties();
    properties["grep"] = json!({"type": "string", "description": "A regular expression; a line is kept when it is found anywhere in it."});
    properties["source"] = json!({"type": "string", "description": "One source — `cluster`, `yenbot`, `yenbot/<app>` — as `log_list` names them; every source when left out."});
    properties["tenant"] = json!({"type": "string", "enum": ["zend", "npcd"], "description": "For the cluster: only one app's lines — `zend`, the coding service, or `npcd`, the bots'."});
    properties["node"] = json!({"type": "string", "description": "For the cluster: one member's node id, as `cluster_status` lists it (Admins only)."});
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Search the logs for what they say about a symptom: every source's matching lines — the \
            cluster's machines, Yen Bots and each of its apps — or one source's, oldest first, each under \
            its source and cut at 400 characters. `grep` is a regular expression found anywhere in a line; \
            narrow with `level`, `since` (UTC), and for the cluster `tenant` and `node`. When more lines \
            match than `limit`, the newest are kept and the answer says earlier ones were left out. Cite \
            the lines you rely on.",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["grep"],
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    if let Err(e) = string(args, "grep") {
        return Answer::error(e);
    }
    let source = match args.get("source").and_then(Value::as_str) {
        Some(name) => match Source::parse(name) {
            Ok(s) => Some(s),
            Err(e) => return Answer::error(e),
        },
        None => None,
    };
    match source {
        Some(Source::Cluster) => cluster_text(call, args).await,
        Some(Source::Site { site, sub }) => {
            let entry = site_entry(call, &site, sub.as_deref(), args).await;
            Answer::ok(render(&json!({"members": [entry]})))
        }
        None => {
            let cluster = cluster_text(call, args).await;
            let mut out = vec![cluster.text];
            match sites(call).await {
                Ok(names) => {
                    let entries =
                        join_all(names.iter().map(|site| site_entry(call, site, None, args))).await;
                    if !entries.is_empty() {
                        out.push(render(&json!({"members": entries})));
                    }
                }
                Err(e) => out.push(format!("(the sites could not be listed: {e})")),
            }
            Answer::ok(out.join("\n\n"))
        }
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}
