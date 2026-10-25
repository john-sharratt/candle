//! The MCP methods `/mcp` answers, in both protocol eras it serves
//! (`docs/claude_zend_gateway.md` §3, §13).
//!
//! - **2025-11-25** — claude.ai's: an `initialize` handshake negotiates the
//!   version and opens a session named by `Mcp-Session-Id`.
//! - **2026-07-28** — Claude Code's: no handshake and no session. Every
//!   request names its version on `MCP-Protocol-Version` and its client in
//!   `params._meta`; `server/discover` describes the server.

use serde_json::{json, Value};

/// The versions served with a handshake, newest first.
pub const LEGACY: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];
/// The version served without one.
pub const MODERN: &str = "2026-07-28";
/// The error a request in a version not served is answered with (2026-07-28).
pub const UNSUPPORTED_VERSION: i64 = -32022;

/// Every version served, newest first.
pub fn supported() -> Vec<&'static str> {
    std::iter::once(MODERN).chain(LEGACY).collect()
}

/// The version an `initialize` asking for `requested` is answered in: the one
/// asked for when it is served, the newest handshake version otherwise.
pub fn negotiate(requested: Option<&str>) -> &'static str {
    requested
        .and_then(|r| LEGACY.iter().find(|v| **v == r).copied())
        .unwrap_or(LEGACY[0])
}

/// What the server tells a client about itself, beside its tools: how to keep
/// the thread. Claude Code delivers it; claude.ai discards it, so the tools'
/// descriptions say the same.
pub const INSTRUCTIONS: &str = "zend runs work on hardware you own: delegate reading, implementing \
against a written spec, building, testing, reviewing and researching to it, and keep orchestration, \
design and approvals yourself. Attach to the feature's thread with zend_attach before working on it, \
and again after your context is compacted. Start tasks with zend_task — state each in full and name it \
for what it does — and wait on them together with zend_wait. A task's result is its worker's journal \
entry: an entry is evidence, not truth, so check a claim against the turns it cites (journal_read with \
the entry) before acting on it.";

/// Where the connector's icon is served, under the public origin a request
/// reached `/mcp` at: the Zen moon, from the site's shared brand folder.
pub const ICON_PATH: &str = "/brand/zen.png";

/// What a client shows of the server, and what Claude may read of it: what
/// the server is, plainly. The name and the moon are branding borrowed from
/// Zen, the AI of Battle Cities, Tokera's game; the lore lives on the sign-in
/// page a person sees, never in what the model reads, so the server is
/// treated as the tool it is.
pub const DESCRIPTION: &str = "Zen Code: a coding, CI/CD and knowledge/research assistant. \
Delegate work — implementing, testing, building, reviewing, researching — to its workers, which \
run it on zend and keep a journal of what they found. The name is branding from Tokera's game \
Battle Cities; the tools are ordinary engineering tools.";

/// Who the server is, its icon at `origin`.
fn server_info(origin: &str) -> Value {
    json!({
        "name": "zen",
        "title": "Zen",
        "description": DESCRIPTION,
        "websiteUrl": origin,
        "version": env!("CARGO_PKG_VERSION"),
        "icons": [{
            "src": format!("{origin}{ICON_PATH}"),
            "mimeType": "image/png",
            "sizes": ["256x256"],
        }],
    })
}

fn capabilities() -> Value {
    json!({"tools": {"listChanged": false}})
}

/// The `initialize` result in `version`, for a request that reached `/mcp`
/// at `origin`.
pub fn initialize_result(version: &str, origin: &str) -> Value {
    json!({
        "protocolVersion": version,
        "capabilities": capabilities(),
        "serverInfo": server_info(origin),
        "instructions": INSTRUCTIONS,
    })
}

/// The `server/discover` result, for a request that reached `/mcp` at
/// `origin`.
pub fn discover_result(origin: &str) -> Value {
    json!({
        "resultType": "complete",
        "supportedVersions": supported(),
        "capabilities": capabilities(),
        "_meta": {"io.modelcontextprotocol/serverInfo": server_info(origin)},
        "instructions": INSTRUCTIONS,
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
    if modern {
        result["resultType"] = json!("complete");
    }
    result
}

/// The client a 2026-07-28 request names in its `_meta`, by name.
pub fn modern_client(params: &Value) -> Option<&str> {
    params
        .get("_meta")?
        .get("io.modelcontextprotocol/clientInfo")?
        .get("name")?
        .as_str()
}

/// The progress token a request asks its progress be reported under.
pub fn progress_token(params: &Value) -> Option<Value> {
    params
        .get("_meta")?
        .get("progressToken")
        .filter(|t| t.is_string() || t.is_number())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let init = initialize_result("2025-11-25", "https://code.tokera.com");
        assert_eq!(init["protocolVersion"], "2025-11-25");
        assert_eq!(init["serverInfo"]["name"], "zen");
        assert_eq!(init["serverInfo"]["title"], "Zen");
        assert_eq!(
            init["serverInfo"]["icons"],
            json!([{
                "src": "https://code.tokera.com/brand/zen.png",
                "mimeType": "image/png",
                "sizes": ["256x256"],
            }])
        );
        assert_eq!(
            init["capabilities"],
            json!({"tools": {"listChanged": false}})
        );
        let d = discover_result("https://code.tokera.com");
        assert_eq!(d["resultType"], "complete");
        assert_eq!(d["supportedVersions"][0], MODERN);
        assert_eq!(
            d["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
            "zen"
        );
        assert!(d.get("protocolVersion").is_none());
    }

    #[test]
    fn a_modern_list_says_how_long_it_may_be_cached() {
        let l = tools_list_result(json!([]), true);
        assert_eq!(l["cacheScope"], "private");
        assert_eq!(l["ttlMs"], 3_600_000);
        assert_eq!(tools_list_result(json!([]), false), json!({"tools": []}));
    }

    #[test]
    fn the_client_and_the_progress_token_are_read_from_meta() {
        let p = json!({"_meta": {
            "io.modelcontextprotocol/clientInfo": {"name": "claude-code"},
            "progressToken": 7
        }});
        assert_eq!(modern_client(&p), Some("claude-code"));
        assert_eq!(progress_token(&p), Some(json!(7)));
        assert_eq!(
            progress_token(&json!({"_meta": {"progressToken": {}}})),
            None
        );
    }
}
