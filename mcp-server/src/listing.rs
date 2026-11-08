//! A server's tool listing, named by a version every answer carries
//! ([`TOOLS_VERSION_HEADER`]), so a server in front of it learns that the
//! listing changed without listing again: the aggregator tells its client
//! with `notifications/tools/list_changed` (`docs/mcp_app.md` §3.1).

use serde_json::{json, Value};

/// The response header every answer names the server's current listing on.
pub const TOOLS_VERSION_HEADER: &str = "x-mcp-tools-version";

/// The version of `tools`, a `tools/list` array: FNV-1a over its JSON, as 16
/// hex digits. Stable across runs and builds for the same listing, which a std
/// hasher does not promise, and different for any change a client would see:
/// a tool added, removed, renamed, or its description or schema changed.
pub fn version(tools: &Value) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in tools.to_string().bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The notification that tells a client its tool list changed, so it lists
/// again.
pub fn list_changed() -> Value {
    json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A listing's version is its content's**: the same listing, the same
    /// version; a description changed, another one.
    #[test]
    fn a_listing_is_versioned_by_its_content() {
        let one = json!([{"name": "file_read", "description": "Read a file."}]);
        assert_eq!(version(&one), "9a1b5b4e7f0bd7a6");
        assert_eq!(version(&one), version(&one.clone()));
        let other = json!([{"name": "file_read", "description": "Read files."}]);
        assert_ne!(version(&one), version(&other));
        assert_eq!(version(&json!([])), "6ea3dabe0dc0cd09");
    }

    #[test]
    fn the_notification_is_the_specs() {
        assert_eq!(
            list_changed().to_string(),
            r#"{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}"#
        );
    }
}
