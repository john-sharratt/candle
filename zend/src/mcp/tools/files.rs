//! `file_read`, `file_list` and `file_search` on the connector
//! (`docs/mcp_file_tools.md` §2): the workers' own tools — the same
//! definitions, the same arguments, the same answers — run over the workspace
//! as it is committed, so what claude.ai reads is what `file_edit_and_commit`
//! edits; or, given `conv`, over the workspace as that conversation left it
//! (§2.1). They only read.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};
use web::auth::Role;
use zend_tools::ToolContext;

use super::{Answer, Ctx, LOADING};
use crate::mcp::chats::dialogue::not_there;
use crate::mcp::ownership::{percent_decode, Viewer};
use crate::session::ConversationTools;
use crate::tool_def;
use crate::tools::{run_tool, ToolCall};

/// The worker tools the connector offers as they are.
pub const NAMES: [&str; 3] = ["file_read", "file_list", "file_search"];

/// What the connector says beside a worker's description: where the files
/// come from.
const FROM: &str = " On the connector these read the repositories as they are committed \
    (their main branch unless a tool names another) — or, given `conv`, as that conversation \
    left them, its uncommitted changes included; `file_edit_and_commit` edits them.";

/// The connector's own argument naming a conversation to read through.
const CONV: &str = "conv";

/// The most files one `file_read` with `paths` reads.
const MAX_PATHS: usize = 64;
/// The most bytes one `file_read` with `paths` answers, all files together.
const MAX_PATHS_BYTES: usize = 4 * 1024 * 1024;

/// What the connector's `file_read` says beside the worker's description:
/// `paths`, which only the connector takes.
const PATHS: &str = " On the connector `file_read` also takes `paths` in place of `path` and \
    `part`/`lines`/`list`: up to 64 files of one repository read whole in one call, each under \
    its path — text as `text`, any other file as `base64` — 4 MiB at most in all; what an app \
    preview loads itself from.";

/// The connector's definition of worker tool `name`: the catalog's, and for
/// `file_read` its `paths` beside it.
pub fn definition(name: &str) -> Value {
    let def = tool_def::find(name).unwrap_or_else(|| panic!("the catalog defines {name}"));
    let mut schema = def.parameters.clone();
    let mut description = format!("{}{FROM}", def.description);
    schema["properties"][CONV] = json!({
        "type": "string",
        "description": "Read as this conversation left them: its id, as the web UI's `#conv=` link names it, or a task's. Only a conversation you may open is read.",
    });
    if name == "file_read" {
        schema["properties"]["paths"] = json!({
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_PATHS,
            "description": "Files to read whole, in one call, each relative to the repository; in place of `path`.",
        });
        if let Some(required) = schema["required"].as_array_mut() {
            required.retain(|r| r != "path");
        }
        description.push_str(PATHS);
    }
    json!({
        "name": def.name,
        "annotations": {"readOnlyHint": true},
        "description": description,
        "inputSchema": schema,
    })
}

/// `paths` of `repo`, each whole: text as `text`, other bytes as `base64`, a
/// missing file as `missing`.
fn read_paths(tools: &ToolContext, repo: &str, paths: &[String]) -> Answer {
    if paths.len() > MAX_PATHS {
        return Answer::error(format!(
            "`paths` names {}; at most {MAX_PATHS}",
            paths.len()
        ));
    }
    let store = match tools.files.repo(repo) {
        Ok(s) => s,
        Err(e) => return Answer::error(e.to_string()),
    };
    let mut total = 0usize;
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let entry = match store.read_bytes(path) {
            Ok(Some(bytes)) => {
                total += bytes.len();
                if total > MAX_PATHS_BYTES {
                    return Answer::error(format!(
                        "the files come to more than {MAX_PATHS_BYTES} bytes; read fewer at once"
                    ));
                }
                match String::from_utf8(bytes) {
                    Ok(text) => json!({"path": path, "text": text}),
                    Err(e) => {
                        let bytes = e.into_bytes();
                        json!({"path": path, "bytes": bytes.len(), "base64": STANDARD.encode(&bytes)})
                    }
                }
            }
            Ok(None) => json!({"path": path, "missing": true}),
            Err(e) => json!({"path": path, "error": e.to_string()}),
        };
        files.push(entry);
    }
    Answer::ok(json!({"repo": repo, "files": files}).to_string())
}

/// The context a call reads through — the repositories as committed, or,
/// `conv` named, as that conversation left them — and the repositories the
/// conversation's changes could not be put back into, each with why.
/// `Ok(None)` while the model is loading.
fn context(ctx: &Ctx, conv: Option<&str>) -> Result<Option<ConversationTools>, String> {
    let Some(raw) = conv else {
        return Ok(ctx
            .session
            .repository_tools(ctx.caller.role)
            .map(|(tools, _)| (tools, Vec::new())));
    };
    let id = percent_decode(raw.trim());
    // Checked as the web routes check it, and refused as not there either
    // way: a refusal never says a conversation exists.
    let listed = || {
        ctx.session
            .list_conversations(true)
            .iter()
            .any(|c| c.id == id)
    };
    if !Viewer::of_caller(&ctx.caller).may_open(&id) || !listed() {
        return Err(not_there(&id));
    }
    match ctx.session.conversation_tools(ctx.caller.role, &id) {
        None => Ok(None),
        Some(Ok(tools)) => Ok(Some(tools)),
        Some(Err(e)) => Err(format!(
            "conversation `{id}`'s changes could not be read: {e}"
        )),
    }
}

/// Run worker tool `name` with `args`.
pub fn call(ctx: &Ctx, name: &str, args: &Value) -> Answer {
    if !ctx.caller.role.at_least(Role::User) {
        return Answer::error("reading the repositories is a user's; you are a guest");
    }
    let conv = match args.get(CONV) {
        None => None,
        Some(Value::String(s)) => Some(s.as_str()),
        Some(_) => return Answer::error("`conv` is a conversation's id"),
    };
    let (tools, unrestored) = match context(ctx, conv) {
        Ok(Some(found)) => found,
        Ok(None) => return Answer::error(LOADING),
        Err(e) => return Answer::error(e),
    };
    // The worker tool takes no `conv`: it reads whatever its context holds.
    let mut args = args.clone();
    if let Some(map) = args.as_object_mut() {
        map.remove(CONV);
    }
    let answer = read(&tools, name, &args);
    if unrestored.is_empty() {
        return answer;
    }
    let mut note = String::from(
        "\n\nThe conversation's changes could not be put back in these repositories, which read as committed:",
    );
    for (repo, why) in unrestored {
        note.push_str(&format!("\n- {repo}: {why}"));
    }
    Answer {
        text: answer.text + &note,
        ..answer
    }
}

/// Run worker tool `name` with `args` against `tools`.
fn read(tools: &ToolContext, name: &str, args: &Value) -> Answer {
    if let (true, Some(paths)) = (name == "file_read", args.get("paths")) {
        let Some(repo) = args.get("repo").and_then(Value::as_str) else {
            return Answer::error("`repo` names the repository the paths are in");
        };
        let paths: Vec<String> = match serde_json::from_value(paths.clone()) {
            Ok(p) => p,
            Err(e) => return Answer::error(format!("`paths` is not a list of paths: {e}")),
        };
        return read_paths(tools, repo, &paths);
    }
    let answer = run_tool(
        tools,
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
        for name in ["file_list", "file_search"] {
            let def = definition(name);
            let catalog = tool_def::find(name).unwrap();
            let mut want = catalog.parameters.clone();
            want["properties"][CONV] = def["inputSchema"]["properties"][CONV].clone();
            assert_eq!(def["name"], name);
            assert_eq!(def["inputSchema"], want);
            assert_eq!(def["inputSchema"]["properties"][CONV]["type"], "string");
            assert_eq!(def["annotations"]["readOnlyHint"], true);
            assert!(def["description"].as_str().unwrap().ends_with(FROM));
        }
    }

    /// **The connector's `file_read` is the catalog's with `paths` beside
    /// it**, and `path` no longer required, since `paths` stands in for it.
    #[test]
    fn the_connectors_file_read_also_takes_paths() {
        let def = definition("file_read");
        let catalog = tool_def::find("file_read").unwrap();
        let mut want = catalog.parameters.clone();
        want["properties"]["paths"] = def["inputSchema"]["properties"]["paths"].clone();
        want["properties"][CONV] = def["inputSchema"]["properties"][CONV].clone();
        if let Some(r) = want["required"].as_array_mut() {
            r.retain(|r| r != "path");
        }
        assert_eq!(def["inputSchema"], want);
        assert_eq!(def["inputSchema"]["properties"]["paths"]["maxItems"], 64);
        assert!(def["description"].as_str().unwrap().ends_with(PATHS));
    }
}
