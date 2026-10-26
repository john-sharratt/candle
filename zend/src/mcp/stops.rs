//! Claude stopping a running worker — `task_stop`
//! (`docs/claude_zend_gateway.md` §6.2).
//!
//! Like a nudge, a stop never interrupts a decoding turn: it is read where the
//! worker's next tool round ends, and that round then closes the loop as the
//! round cap does — its results carry the stop, no further call runs, and the
//! next turn answers with what the worker has. The worker settles stuck, live,
//! so `task_reply` redirects it. A stop the turn never reaches, because it
//! ended without another round, is cleared when the worker settles.
//!
//! **A stop is a record**, under the worker's conversation, so it reaches the
//! worker whichever member took the call and whichever runs the turn.

use serde_json::{json, Value};

use super::records::Records;
use crate::tools::ToolResult;

fn key(conv_id: &str) -> String {
    format!("mcp/stop/{conv_id}")
}

/// Ask `conv_id`'s next tool round to close its loop, for `reason`.
pub fn request(records: &dyn Records, conv_id: &str, reason: &str) -> anyhow::Result<()> {
    records.put(&key(conv_id), json!({ "reason": reason }))
}

/// The stop asked of `conv_id`, if any, taken: its reason.
pub fn take(records: &dyn Records, conv_id: &str) -> Option<String> {
    let key = key(conv_id);
    let stop = records.get(&key)?;
    records.cease(&key).ok()?;
    Some(
        stop.get("reason")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    )
}

/// Mark the round a stop closes, on its last result: the worker reads that
/// Claude stopped it, why, and that it now answers with what it has.
pub fn notice(results: &mut [ToolResult], reason: &str) {
    let Some(last) = results.last_mut() else {
        return;
    };
    let why = if reason.trim().is_empty() {
        String::new()
    } else {
        format!(" Claude's reason: {}.", reason.trim().trim_end_matches('.'))
    };
    let notice = Value::String(format!(
        "Claude stopped this task: no further tool call will run.{why} Answer now with what you \
         have — what you found, and what remains undone — so Claude can redirect you."
    ));
    match &mut last.response {
        Value::Object(map) => {
            map.insert("notice".to_string(), notice);
        }
        other => {
            let result = other.take();
            *other = json!({"result": result, "notice": notice});
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;
    use crate::tools::ToolCall;

    #[test]
    fn a_stop_is_taken_once_with_its_reason() {
        let r = MemRecords::default();
        assert_eq!(take(&r, "mcp:o:t/a"), None);
        request(&r, "mcp:o:t/a", "use v2 only").unwrap();
        assert_eq!(take(&r, "mcp:o:t/ab"), None, "another worker's is its own");
        assert_eq!(take(&r, "mcp:o:t/a").as_deref(), Some("use v2 only"));
        assert_eq!(take(&r, "mcp:o:t/a"), None);
    }

    #[test]
    fn the_closing_round_says_claude_stopped_it_and_why() {
        let result = |response: Value| ToolResult {
            call: ToolCall {
                name: "web_fetch".into(),
                arguments: json!({}),
            },
            response,
        };
        let mut results = vec![result(json!({"status": 404}))];
        notice(&mut results, "use v2 only.");
        assert_eq!(
            results[0].response,
            json!({
                "status": 404,
                "notice": "Claude stopped this task: no further tool call will run. Claude's \
                           reason: use v2 only. Answer now with what you have — what you found, \
                           and what remains undone — so Claude can redirect you."
            })
        );
        let mut bare = vec![result(json!("text"))];
        notice(&mut bare, "");
        assert_eq!(bare[0].response["result"], "text");
        assert!(bare[0].response["notice"]
            .as_str()
            .unwrap()
            .starts_with("Claude stopped this task: no further tool call will run. Answer now"));
    }
}
