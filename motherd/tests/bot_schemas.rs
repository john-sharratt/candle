//! **Zen's bot and picture tools ask npcd for nothing npcd does not take**
//! (`docs/zen_tool_reach.md` §4.3). Zen reaches the bots through npcd's own
//! MCP tools, and its definitions of them are written in zend's catalog; this
//! holds each one inside npcd's: every argument Zen offers is one npcd's
//! schema names, of the same type, and the two require the same arguments. A
//! change to npcd's schema that Zen's no longer fits fails here.

use std::collections::BTreeSet;

use serde_json::Value;

fn names(schema: &Value, key: &str) -> BTreeSet<String> {
    match &schema[key] {
        Value::Object(m) => m.keys().cloned().collect(),
        Value::Array(a) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => BTreeSet::new(),
    }
}

#[test]
fn zens_bot_tools_fit_inside_npcds_own() {
    let npcd = npcd::mcp::definitions();
    let mut fitted = 0;
    for def in npcd.as_array().expect("npcd's tools are a list") {
        let name = def["name"].as_str().expect("a tool has a name");
        let Some(zen) = zend::tool_def::find(name) else {
            continue;
        };
        fitted += 1;
        let theirs = &def["inputSchema"];
        let ours = &zen.parameters;
        let offered = names(ours, "properties");
        let taken = names(theirs, "properties");
        assert!(
            offered.is_subset(&taken),
            "{name}: Zen offers {:?}, which npcd does not take",
            offered.difference(&taken).collect::<Vec<_>>()
        );
        assert_eq!(
            names(ours, "required"),
            names(theirs, "required"),
            "{name}: Zen and npcd require different arguments"
        );
        for field in &offered {
            if let Some(ty) = theirs["properties"][field].get("type") {
                assert_eq!(
                    ours["properties"][field].get("type"),
                    Some(ty),
                    "{name}: `{field}` is a {ty} to npcd"
                );
            }
        }
    }
    assert_eq!(
        fitted, 13,
        "every bot and picture tool but image_upload is Zen's"
    );
}
