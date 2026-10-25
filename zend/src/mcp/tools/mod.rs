//! The tools `/mcp` offers (`docs/claude_zend_gateway.md` §3.2), one file
//! each: its definition — name, description, input schema — and its call.
//!
//! **The descriptions carry the rules.** claude.ai discards a server's
//! `instructions` (§13.3), so what Claude must do to keep its thread — attach
//! first, name each task for what it is for, read the journal after its
//! context is compacted — is said in the descriptions, the one channel every
//! client delivers.

use std::sync::Arc;

use serde_json::{json, Value};

use super::caller::Caller;
use super::state::{Session, DEFAULT_WAIT_S};
use crate::session::ZendSession;

mod answer;
mod archive;
mod attach;
mod journal_grep;
mod journal_note;
mod journal_read;
mod nudge;
mod task;
mod threads;
mod wait;

/// What a tool answers when it needs the model and zend is still loading it.
pub const LOADING: &str = "zend is still loading its model; try again shortly";

/// What a tool call runs with.
pub struct Ctx {
    pub session: Arc<ZendSession>,
    pub caller: Caller,
    /// The MCP session the call came on, and its id.
    pub mcp: Session,
    pub mcp_id: String,
}

/// A tool's answer: its text, and whether it is an error the model should
/// read as one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub text: String,
    pub is_error: bool,
}

impl Answer {
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_error: true,
        }
    }

    /// The `tools/call` result: always a text block, which every client reads.
    pub fn result(&self) -> Value {
        json!({
            "content": [{"type": "text", "text": self.text}],
            "isError": self.is_error,
        })
    }
}

/// Every tool's definition, for `tools/list`.
pub fn definitions() -> Value {
    json!([
        attach::definition(),
        task::definition(),
        wait::definition(),
        answer::definition(),
        nudge::definition(),
        journal_read::definition(),
        journal_grep::definition(),
        journal_note::definition(),
        threads::definition(),
        archive::definition(),
    ])
}

/// Run tool `name` with `args`.
pub async fn call(ctx: &Ctx, name: &str, args: &Value) -> Answer {
    match name {
        attach::NAME => attach::call(ctx, args),
        task::NAME => task::call(ctx, args).await,
        wait::NAME => wait::call(ctx, args).await,
        answer::NAME => answer::call(ctx, args).await,
        nudge::NAME => nudge::call(ctx, args),
        journal_read::NAME => journal_read::call(ctx, args),
        journal_note::NAME => journal_note::call(ctx, args),
        journal_grep::NAME => journal_grep::call(ctx, args),
        threads::NAME => threads::call(ctx, args),
        archive::NAME => archive::call(ctx, args),
        other => Answer::error(format!("there is no tool `{other}`")),
    }
}

/// A string argument, refused when missing or empty.
pub fn string<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("`{key}` is missing: it is a non-empty string"))
}

/// The seconds a call may wait: `wait_s` if given, the default otherwise, at
/// most what the session's client allows.
pub fn wait_secs(ctx: &Ctx, args: &Value) -> u64 {
    let ceiling = ctx.mcp.client.wait_ceiling();
    args.get("wait_s")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_WAIT_S)
        .min(ceiling)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_a_name_a_description_and_an_object_schema() {
        let defs = definitions();
        let names: Vec<&str> = defs
            .as_array()
            .unwrap()
            .iter()
            .map(|d| {
                assert!(d["description"].as_str().unwrap().len() > 40, "{d}");
                assert_eq!(d["inputSchema"]["type"], "object", "{d}");
                d["name"].as_str().unwrap()
            })
            .collect();
        assert_eq!(
            names,
            [
                "zend_attach",
                "zend_task",
                "zend_wait",
                "zend_answer",
                "zend_nudge",
                "journal_read",
                "journal_grep",
                "journal_note",
                "zend_threads",
                "zend_archive"
            ]
        );
    }

    #[test]
    fn an_answer_is_a_text_block() {
        assert_eq!(
            Answer::error("no").result().to_string(),
            r#"{"content":[{"type":"text","text":"no"}],"isError":true}"#
        );
    }

    #[test]
    fn a_string_argument_is_trimmed_and_an_empty_one_refused() {
        let args = json!({"a": "  x ", "b": "  "});
        assert_eq!(string(&args, "a").unwrap(), "x");
        assert!(string(&args, "b").is_err());
        assert!(string(&args, "c").is_err());
    }
}
