//! `log_list`: every log source — the cluster, and each site's sources with
//! how many lines each holds and its newest line's time.

use axum::http::Method;
use serde_json::{json, Value};

use super::{sites, CLUSTER, WITHIN};
use crate::mcp::ops::json_of;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{Answer, Ctx};

pub const NAME: &str = "log_list";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "List every log you can read: `cluster` — the machines running the user's daemons \
            (motherd, with its zend and npcd tenants) — and each site that keeps its own, such as `yenbot` \
            for Yen Bots, with one source per app (`yenbot/<app>`: its browser errors, workflow runs and \
            check refusals). Each with how many lines it holds and when it last wrote. Read one with \
            `log_read`, search them with `log_grep`.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
    })
}

/// One site's sources, as `log_list` prints them.
pub fn render_site(site: &str, sources: &Value) -> String {
    let rows: Vec<String> = sources["sources"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| {
            let name = s["name"].as_str().unwrap_or("?");
            let lines = s["lines"].as_u64().unwrap_or(0);
            match s["newest"].as_str() {
                Some(at) => format!("{name} — {lines} lines, newest {at}"),
                None => format!("{name} — {lines} lines"),
            }
        })
        .collect();
    if rows.is_empty() {
        format!("{site} — no sources")
    } else {
        rows.join("\n")
    }
}

pub async fn run(call: &OpsCall) -> Answer {
    let mut out = vec![format!(
        "{CLUSTER} — every member's daemon log (`tenant` narrows to zend or npcd)"
    )];
    let names = match sites(call).await {
        Ok(n) => n,
        Err(e) => {
            out.push(format!("(the sites could not be listed: {e})"));
            return Answer::ok(out.join("\n"));
        }
    };
    for site in names {
        let path = format!("/gateway/logs/{site}/v1/logs/sources");
        let row = match call.up(Method::GET, &path, None, WITHIN).await {
            Ok(r) if r.status == 200 => match json_of(&r) {
                Ok(v) => render_site(&site, &v),
                Err(e) => format!("{site} — {e}"),
            },
            Ok(r) => format!("{site} — answered {}", r.status),
            Err(e) => format!("{site} — no answer: {e}"),
        };
        out.push(row);
    }
    Answer::ok(out.join("\n"))
}

pub async fn call(ctx: &Ctx, _args: &Value) -> Answer {
    run(&OpsCall::of(ctx)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sites_sources_are_listed_with_their_size_and_last_line() {
        let sources = json!({"sources": [
            {"name": "yenbot", "lines": 120, "newest": "2026-11-06T10:02:11Z"},
            {"name": "yenbot/recipes", "lines": 3},
        ]});
        assert_eq!(
            render_site("yenbot", &sources),
            "yenbot — 120 lines, newest 2026-11-06T10:02:11Z\nyenbot/recipes — 3 lines"
        );
        assert_eq!(render_site("yenbot", &json!({})), "yenbot — no sources");
    }
}
