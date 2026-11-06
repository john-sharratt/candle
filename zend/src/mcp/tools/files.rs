//! `file_read`, `file_list` and `file_search` on the connector
//! (`docs/mcp_file_tools.md` §2): the workers' own tools — the same
//! definitions, the same arguments, the same answers — run over the workspace
//! as it is committed, so what claude.ai reads is what `file_edit_and_commit`
//! edits. They only read.

use serde_json::{json, Value};
use web::auth::Role;

use super::{Answer, Ctx, LOADING};
use crate::tool_def;
use crate::tools::{run_tool, ToolCall};

/// The worker tools the connector offers as they are.
pub const NAMES: [&str; 3] = ["file_read", "file_list", "file_search"];

/// What the connector says beside a worker's description: where the files
/// come from.
const FROM: &str = " On the connector these read the repositories as they are committed \
    (their main branch unless a tool names another), never a conversation's own changes; \
    `file_edit_and_commit` edits them.";

/// The connector's definition of worker tool `name`.
pub fn definition(name: &str) -> Value {
    let def = tool_def::find(name).unwrap_or_else(|| panic!("the catalog defines {name}"));
    json!({
        "name": def.name,
        "annotations": {"readOnlyHint": true},
        "description": format!("{}{FROM}", def.description),
        "inputSchema": def.parameters,
    })
}

/// Run worker tool `name` with `args`.
pub fn call(ctx: &Ctx, name: &str, args: &Value) -> Answer {
    if !ctx.caller.role.at_least(Role::User) {
        return Answer::error("reading the repositories is a user's; you are a guest");
    }
    let Some((tools, _)) = ctx.session.repository_tools() else {
        return Answer::error(LOADING);
    };
    let answer = run_tool(
        &tools,
        &ToolCall {
            name: name.to_string(),
            arguments: args.clone(),
        },
    );
    let failed = answer.get("error").is_some();
    let text = serde_json::to_string(&answer).unwrap_or_else(|e| e.to_string());
    if failed {
        Answer::error(text)
    } else {
        Answer::ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The connector shows each worker tool as the catalog defines it**,
    /// read-only, with where its files come from.
    #[test]
    fn each_file_tool_is_the_catalogs() {
        for name in NAMES {
            let def = definition(name);
            let catalog = tool_def::find(name).unwrap();
            assert_eq!(def["name"], name);
            assert_eq!(def["inputSchema"], catalog.parameters);
            assert_eq!(def["annotations"]["readOnlyHint"], true);
            assert!(def["description"].as_str().unwrap().ends_with(FROM));
        }
    }
}
