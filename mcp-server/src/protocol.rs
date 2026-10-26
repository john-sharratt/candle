//! The MCP methods a server answers, in both protocol eras it serves
//! (`docs/claude_zend_gateway.md` §3, §13).
//!
//! - **2025-11-25** — claude.ai's: an `initialize` handshake negotiates the
//!   version and opens a session named by `Mcp-Session-Id`.
//! - **2026-07-28** — Claude Code's: no handshake and no session. Every
//!   request names its version on `MCP-Protocol-Version` and its client in
//!   `params._meta`; `server/discover` describes the server.

use std::iter;

use serde_json::{json, Value};

use crate::server::Identity;

/// The versions served with a handshake, newest first.
pub const LEGACY: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];
/// The version served without one.
pub const MODERN: &str = "2026-07-28";
/// The error a request in a version not served is answered with (2026-07-28).
pub const UNSUPPORTED_VERSION: i64 = -32022;

/// Where a 2026-07-28 request names its client.
pub const CLIENT_META: &str = "io.modelcontextprotocol/clientInfo";
/// Where a request from a server calling on its client's behalf names that
/// server's own session, so what is held for the client is held per session
/// of the server in front (`docs/mcp_app.md` §3).
pub const SESSION_META: &str = "com.tokera/session";

/// Every version served, newest first.
pub fn supported() -> Vec<&'static str> {
    iter::once(MODERN).chain(LEGACY).collect()
}

/// The version an `initialize` asking for `requested` is answered in: the one
/// asked for when it is served, the newest handshake version otherwise.
pub fn negotiate(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|r| LEGACY.iter().find(|v| **v == r).copied())
        .unwrap_or(LEGACY[0])
}

/// Who the server is, its icon at `origin`.
pub fn server_info(identity: &Identity, origin: &str) -> Value {
    let mut info = json!({
        "name": identity.name,
        "title": identity.title,
        "description": identity.description,
        "websiteUrl": origin,
        "version": identity.version,
    });
    if let Some(icon) = &identity.icon {
        info["icons"] = json!([{
            "src": format!("{origin}{icon}"),
            "mimeType": "image/png",
            "sizes": ["256x256"],
        }]);
    }
    info
}

fn capabilities() -> Value {
    json!({"tools": {"listChanged": false}})
}

/// The `initialize` result in `version`, for a request that reached the server
/// at `origin`.
pub fn initialize_result(identity: &Identity, version: &str, origin: &str) -> Value {
    json!({
        "protocolVersion": version,
        "capabilities": capabilities(),
        "serverInfo": server_info(identity, origin),
        "instructions": identity.instructions,
    })
}

/// The `server/discover` result, for a request that reached the server at
/// `origin`.
pub fn discover_result(identity: &Identity, origin: &str) -> Value {
    json!({
        "resultType": "complete",
        "supportedVersions": supported(),
        "capabilities": capabilities(),
        "_meta": {"io.modelcontextprotocol/serverInfo": server_info(identity, origin)},
        "instructions": identity.instructions,
    })
}

/// The `tools/list` result over `tools`. In 2026-07-28 a list result says how
/// long it may be cached and by whom: an hour, by this caller alone, since
/// what a tool may do depends on who asks.
pub fn tools_list_result(tools: Value, modern: bool) -> Value {
    if modern {
        json!({"resultType": "complete", "tools": tools, "ttlMs": 3_600_000, "cacheScope": "private"})
    } else {
        json!({"tools": tools})
    }
}

/// A `tools/call` result in its era.
pub fn call_result(mut result: Value, modern: bool) -> Value {
    match result.as_object_mut() {
        Some(fields) if modern => {
            fields.insert("resultType".into(), json!("complete"));
        }
        Some(fields) => {
            fields.remove("resultType");
        }
        None => {}
    }
    result
}

/// A `tools/call` result carrying `text`, and whether it is an error the model
/// reads as one: a text block, which every client reads.
pub fn text_result(text: &str, is_error: bool) -> Value {
    json!({
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    })
}

fn meta<'a>(params: &'a Value, key: &str) -> Option<&'a Value> {
    params.get("_meta")?.get(key)
}

/// The client a 2026-07-28 request names in its `_meta`, by name.
pub fn modern_client(params: &Value) -> Option<&str> {
    meta(params, CLIENT_META)?.get("name")?.as_str()
}

/// The session of the server a request comes from, calling on its client's
/// behalf.
pub fn on_behalf(params: &Value) -> Option<&str> {
    meta(params, SESSION_META)?
        .as_str()
        .filter(|s| !s.is_empty())
}

/// The progress token a request asks its progress be reported under.
pub fn progress_token(params: &Value) -> Option<Value> {
    meta(params, "progressToken")
        .filter(|t| t.is_string() || t.is_number())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zen() -> Identity {
        Identity {
            name: "zen".into(),
            title: "Zen".into(),
            description: "A coding assistant.".into(),
            instructions: "Attach first.".into(),
            icon: Some("/brand/zen.png".into()),
            version: "1.0.0".into(),
        }
    }

    #[test]
    fn a_served_version_is_kept_and_any_other_gets_the_newest_handshake() {
        assert_eq!(negotiate(Some("2025-06-18")), "2025-06-18");
        assert_eq!(negotiate(Some("2024-11-05")), "2025-11-25");
        assert_eq!(negotiate(Some(MODERN)), "2025-11-25");
        assert_eq!(negotiate(None), "2025-11-25");
        assert_eq!(supported()[0], MODERN);
    }

    #[test]
    fn the_handshake_and_discovery_results_are_shaped_as_their_specs_say() {
        let init = initialize_result(&zen(), "2025-11-25", "https://code.tokera.com");
        assert_eq!(
            init,
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {
                    "name": "zen",
                    "title": "Zen",
                    "description": "A coding assistant.",
                    "websiteUrl": "https://code.tokera.com",
                    "version": "1.0.0",
                    "icons": [{
                        "src": "https://code.tokera.com/brand/zen.png",
                        "mimeType": "image/png",
                        "sizes": ["256x256"],
                    }],
                },
                "instructions": "Attach first.",
            })
        );
        let d = discover_result(&zen(), "https://code.tokera.com");
        assert_eq!(d["resultType"], "complete");
        assert_eq!(d["supportedVersions"][0], MODERN);
        assert_eq!(
            d["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
            "zen"
        );
        assert!(d.get("protocolVersion").is_none());
    }

    #[test]
    fn a_server_with_no_icon_names_none() {
        let mut plain = zen();
        plain.icon = None;
        assert!(server_info(&plain, "https://x").get("icons").is_none());
    }

    #[test]
    fn a_modern_list_says_how_long_it_may_be_cached() {
        let l = tools_list_result(json!([]), true);
        assert_eq!(l["cacheScope"], "private");
        assert_eq!(l["ttlMs"], 3_600_000);
        assert_eq!(tools_list_result(json!([]), false), json!({"tools": []}));
    }

    #[test]
    fn a_call_result_is_shaped_for_its_era() {
        let text = text_result("no", true);
        assert_eq!(
            text.to_string(),
            r#"{"content":[{"type":"text","text":"no"}],"isError":true}"#
        );
        let modern = call_result(text.clone(), true);
        assert_eq!(modern["resultType"], "complete");
        assert_eq!(call_result(modern, false), text);
    }

    #[test]
    fn the_client_the_callers_session_and_the_progress_token_are_read_from_meta() {
        let p = json!({"_meta": {
            "io.modelcontextprotocol/clientInfo": {"name": "claude-code"},
            "com.tokera/session": "s1",
            "progressToken": 7
        }});
        assert_eq!(modern_client(&p), Some("claude-code"));
        assert_eq!(on_behalf(&p), Some("s1"));
        assert_eq!(progress_token(&p), Some(json!(7)));
        assert_eq!(
            progress_token(&json!({"_meta": {"progressToken": {}}})),
            None
        );
        assert_eq!(
            on_behalf(&json!({"_meta": {"com.tokera/session": ""}})),
            None
        );
    }
}
