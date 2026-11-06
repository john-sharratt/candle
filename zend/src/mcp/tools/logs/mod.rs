//! The connector's log tools (`docs/mcp_file_tools.md` §6): `log_list`,
//! `log_read` and `log_grep` over every **log source** — the cluster's
//! members, gathered by the gateway as `cluster_logs` gathers them for a
//! worker, and each site the gateway lists as one (`/gateway/logs`), with the
//! sub-sources its `/v1/logs/sources` names: Yen Bots is `yenbot` and one
//! `yenbot/<app>` per app.
//!
//! A source is named `cluster`, `<site>` or `<site>/<sub>`. Every read goes
//! through the gateway as the person whose connector asks, so a site answers
//! only what that person may read.

pub mod grep;
pub mod list;
pub mod read;

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};
use url::form_urlencoded::Serializer;

use super::cluster_logs;
use crate::mcp::ops::json_of;
use crate::mcp::ops_tools::OpsCall;

/// The cluster's members, as a source.
pub const CLUSTER: &str = "cluster";

/// How long a site has to answer.
const WITHIN: Duration = Duration::from_secs(30);

/// Where a named source is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Cluster,
    /// A site, and the sub-source within it when one is named.
    Site {
        site: String,
        sub: Option<String>,
    },
}

impl Source {
    pub fn parse(name: &str) -> Result<Source, String> {
        let name = name.trim();
        if name == CLUSTER {
            return Ok(Source::Cluster);
        }
        let site = name.split('/').next().unwrap_or_default();
        if site.is_empty() || !site.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return Err(format!(
                "`{name}` is not a source: `cluster`, a site such as `yenbot`, or `yenbot/<app>` (log_list names them)"
            ));
        }
        Ok(Source::Site {
            site: site.to_string(),
            sub: (name.len() > site.len()).then(|| name.to_string()),
        })
    }
}

/// The sites the gateway lists as log sources.
pub async fn sites(call: &OpsCall) -> Result<Vec<String>, String> {
    let reply = call.up(Method::GET, "/gateway/logs", None, WITHIN).await?;
    let listed = json_of(&reply)?;
    Ok(listed["sites"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| s.as_str().map(str::to_string))
        .collect())
}

/// The site search `args` asks of `site` for `sub`.
pub fn site_path(site: &str, sub: Option<&str>, args: &Value) -> String {
    let mut q = Serializer::new(String::new());
    for key in ["grep", "level", "since"] {
        if let Some(v) = args.get(key).and_then(Value::as_str) {
            q.append_pair(key, v);
        }
    }
    if let Some(sub) = sub {
        q.append_pair("source", sub);
    }
    let limit = cluster_logs::limit_of(args);
    q.append_pair("limit", &limit.to_string());
    format!("/gateway/logs/{site}/v1/logs?{}", q.finish())
}

/// One site's lines, as an entry `cluster_logs::render` prints under its
/// source.
pub async fn site_entry(call: &OpsCall, site: &str, sub: Option<&str>, args: &Value) -> Value {
    let reply = call
        .up(Method::GET, &site_path(site, sub, args), None, WITHIN)
        .await;
    let (status, body) = match reply {
        Ok(r) => (
            r.status,
            serde_json::from_slice::<Value>(&r.body).unwrap_or(Value::Null),
        ),
        Err(_) => (502, Value::Null),
    };
    json!({
        "host": sub.unwrap_or(site),
        "node": "(site)",
        "status": status,
        "body": body,
    })
}

/// The cluster's lines for `args`, as the worker's `cluster_logs` reads them.
pub async fn cluster_text(call: &OpsCall, args: &Value) -> super::Answer {
    let mut args = args.clone();
    if let Some(fields) = args.as_object_mut() {
        fields.remove("source");
    }
    cluster_logs::run(call, &args).await
}

/// The JSON-schema properties every search takes.
pub fn search_properties() -> Value {
    json!({
        "level": {"type": "string", "enum": ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"], "description": "The least level kept."},
        "since": {"type": "string", "description": "Only lines stamped at or after this: an RFC 3339 prefix such as `2026-11-06T09:20` (UTC)."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5000, "description": "The most lines per member or source, the newest kept (100 by default)."}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_source_is_the_cluster_a_site_or_one_of_its_sources() {
        assert_eq!(Source::parse("cluster"), Ok(Source::Cluster));
        assert_eq!(
            Source::parse("yenbot"),
            Ok(Source::Site {
                site: "yenbot".into(),
                sub: None
            })
        );
        assert_eq!(
            Source::parse("yenbot/recipes"),
            Ok(Source::Site {
                site: "yenbot".into(),
                sub: Some("yenbot/recipes".into())
            })
        );
        assert!(Source::parse("../x").is_err());
        assert!(Source::parse("").is_err());
    }

    #[test]
    fn a_site_search_carries_its_source_and_query() {
        assert_eq!(
            site_path(
                "yenbot",
                Some("yenbot/recipes"),
                &json!({"grep": "Type Error", "level": "WARN", "limit": 9999})
            ),
            "/gateway/logs/yenbot/v1/logs?grep=Type+Error&level=WARN&source=yenbot%2Frecipes&limit=5000"
        );
        assert_eq!(
            site_path("yenbot", None, &json!({})),
            "/gateway/logs/yenbot/v1/logs?limit=100"
        );
    }
}
