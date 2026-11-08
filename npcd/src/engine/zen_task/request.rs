//! What a drone's task act asks of zend, and what zend's answer means
//! (`docs/mission_templates.md` §7): each act is one `tools/call` to zend's
//! `/v1/mcp`, made as Zen (`service:zen`), on the thread named for the mission.

use serde_json::{json, Map, Value};

/// The client a drone's calls name.
pub const CLIENT: &str = "npcd";

/// Where zend's MCP server is, as a member reaches it.
pub const ZEND_MCP: &str = "/v1/mcp";

/// The thread a carried mission's task runs on: one per mission, by its
/// carrier and the chapter it is carried in.
pub fn thread_for(npc: u64, chapter: u64) -> String {
    format!("mission-{npc}-{chapter}")
}

/// The zend tool a task act calls, and its arguments: `task_start` on the
/// mission's `thread`; `task_reply` and `task_nudge` to the task already
/// started, `handle`.
pub fn zend_call(
    act: &str,
    args: &Map<String, Value>,
    thread: &str,
    handle: Option<&str>,
) -> Result<(&'static str, Value), String> {
    let text = |key: &str| {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .ok_or(format!("Say the `{key}`."))
    };
    let started = || {
        handle
            .ok_or_else(|| "Your mission has no task yet: start it with `task_start`.".to_string())
    };
    match act {
        "task_start" => {
            if handle.is_some() {
                return Err("Your mission's task is started already: answer or steer it.".into());
            }
            let mut call = json!({
                "thread": thread,
                "name": text("name")?,
                "prompt": text("prompt")?,
                "driven": "attended",
                "wait_s": 0,
            });
            if let Ok(role) = text("role") {
                call["role"] = json!(role);
            }
            Ok(("repo_task_start", call))
        }
        "task_reply" => Ok((
            "repo_task_reply",
            json!({"task": started()?, "answer": text("answer")?, "wait_s": 0}),
        )),
        "task_nudge" => Ok((
            "repo_task_nudge",
            json!({"task": started()?, "message": text("message")?}),
        )),
        other => Err(format!("`{other}` is no act that drives a task.")),
    }
}

/// How a task stands, as `job_status` says it.
pub fn status_call(handle: &str) -> (&'static str, Value) {
    ("job_status", json!({"jobs": [handle], "wait_s": 0}))
}

/// The JSON-RPC `tools/call` for `tool` with `args`, on behalf of the
/// mission's `thread` — the session zend keeps what it holds for the call
/// under.
pub fn rpc(tool: &str, args: &Value, thread: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": args,
            "_meta": {
                "io.modelcontextprotocol/clientInfo": {"name": CLIENT},
                "com.tokera/session": thread,
            },
        },
    })
}

/// A `tools/call` answer's text, `Err` with it when zend answered an error:
/// the body itself, or the last message of an event stream that carries a
/// result or an error.
pub fn answer_of(body: &[u8]) -> Result<String, String> {
    let text = String::from_utf8_lossy(body);
    let message = serde_json::from_str::<Value>(text.trim()).ok().or_else(|| {
        text.lines()
            .rev()
            .filter_map(|l| l.strip_prefix("data:"))
            .filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok())
            .find(|m| m.get("result").is_some() || m.get("error").is_some())
    });
    let Some(message) = message else {
        return Err(format!("zend answered nothing readable: {}", text.trim()));
    };
    if let Some(e) = message.get("error") {
        return Err(e
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| e.to_string(), str::to_string));
    }
    let result = &message["result"];
    let said = result["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    match result["isError"].as_bool() {
        Some(true) => Err(said),
        _ => Ok(said),
    }
}

/// The task handle a start answered with, `task:<thread>/<name>`.
pub fn handle_in(text: &str, thread: &str) -> Option<String> {
    let at = text.find(&format!("task:{thread}/"))?;
    let handle: String = text[at..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '/' | '-' | '.' | '_'))
        .collect();
    Some(handle.trim_end_matches('.').to_string())
}

/// Where a task stands, from `job_status`'s line for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    Running,
    Asking,
    Stuck,
    Done,
    Failed,
}

/// Where the task `handle` stands, as `job_status`'s answer `text` words it.
pub fn standing_of(text: &str, handle: &str) -> Standing {
    let Some(rest) = text
        .find(handle)
        .map(|at| text[at + handle.len()..].trim_start_matches([' ', '—', '-']))
    else {
        return Standing::Running;
    };
    if rest.starts_with("done") {
        Standing::Done
    } else if rest.starts_with("asks") {
        Standing::Asking
    } else if rest.starts_with("stuck") || rest.starts_with("stopped") {
        Standing::Stuck
    } else if rest.starts_with("error") || rest.starts_with("no such worker") {
        Standing::Failed
    } else {
        Standing::Running
    }
}

/// A digest of what the task said, so the same news is posted once.
pub fn digest(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The most of a task's news one message carries.
pub const NEWS_CHARS: usize = 2_000;

/// The task's news as its carrier reads it on the handset: what `job_status`
/// said, cut at [`NEWS_CHARS`] on a character.
pub fn news(text: &str) -> String {
    let text = text.trim();
    match text.char_indices().nth(NEWS_CHARS) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    /// **Each act is one zend tool, on the mission's thread**, with its
    /// arguments exactly; a reply or nudge before the task, or a second
    /// start, is refused.
    #[test]
    fn each_act_is_one_zend_call() {
        let t = thread_for(7, 3);
        assert_eq!(t, "mission-7-3");
        assert_eq!(
            zend_call(
                "task_start",
                &args(json!({"name": "fix-door", "prompt": "Fix the door.", "role": "coder"})),
                &t,
                None
            )
            .unwrap(),
            (
                "repo_task_start",
                json!({"thread": "mission-7-3", "name": "fix-door", "prompt": "Fix the door.",
                       "driven": "attended", "wait_s": 0, "role": "coder"})
            )
        );
        let h = Some("task:mission-7-3/fix-door");
        assert_eq!(
            zend_call("task_reply", &args(json!({"answer": "Replace it."})), &t, h).unwrap(),
            (
                "repo_task_reply",
                json!({"task": "task:mission-7-3/fix-door", "answer": "Replace it.", "wait_s": 0})
            )
        );
        assert_eq!(
            zend_call(
                "task_nudge",
                &args(json!({"message": "Look at the sensor."})),
                &t,
                h
            )
            .unwrap(),
            (
                "repo_task_nudge",
                json!({"task": "task:mission-7-3/fix-door", "message": "Look at the sensor."})
            )
        );
        assert!(
            zend_call("task_reply", &args(json!({"answer": "x"})), &t, None)
                .unwrap_err()
                .contains("no task yet")
        );
        assert!(zend_call(
            "task_start",
            &args(json!({"name": "a", "prompt": "b"})),
            &t,
            h
        )
        .unwrap_err()
        .contains("started already"));
        assert_eq!(
            zend_call("task_start", &args(json!({"name": "a"})), &t, None).unwrap_err(),
            "Say the `prompt`."
        );
    }

    /// **The call names npcd and the mission's thread as its session.**
    #[test]
    fn the_rpc_names_its_client_and_session() {
        let r = rpc("job_status", &json!({"jobs": ["task:t/n"]}), "mission-7-3");
        assert_eq!(r["method"], "tools/call");
        assert_eq!(r["params"]["name"], "job_status");
        assert_eq!(
            r["params"]["_meta"]["io.modelcontextprotocol/clientInfo"]["name"],
            "npcd"
        );
        assert_eq!(r["params"]["_meta"]["com.tokera/session"], "mission-7-3");
    }

    /// **An answer is its text, from a body or the stream's last message**;
    /// an error, or a result marked an error, is `Err` with its words.
    #[test]
    fn an_answer_is_read_from_a_body_or_a_stream() {
        let ok = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Started task:mission-7-3/fix-door."}],"isError":false}}"#;
        assert_eq!(answer_of(ok).unwrap(), "Started task:mission-7-3/fix-door.");
        let streamed = "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"done\"}]}}\n\n";
        assert_eq!(answer_of(streamed.as_bytes()).unwrap(), "done");
        let refused = br#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"no such thread"}],"isError":true}}"#;
        assert_eq!(answer_of(refused).unwrap_err(), "no such thread");
        let error = br#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"unknown tool"}}"#;
        assert_eq!(answer_of(error).unwrap_err(), "unknown tool");
    }

    /// **The handle is read from the start's answer**, its sentence's full
    /// stop left out.
    #[test]
    fn the_handle_is_read_from_the_start() {
        assert_eq!(
            handle_in(
                "Started task:mission-7-3/fix-door.2. It runs on its own: `job_status`…",
                "mission-7-3"
            ),
            Some("task:mission-7-3/fix-door.2".into())
        );
        assert_eq!(handle_in("nothing started", "mission-7-3"), None);
    }

    /// **Where a task stands is read from its line**, each settled state by
    /// its word.
    #[test]
    fn a_tasks_standing_is_read_from_its_line() {
        let h = "task:mission-7-3/fix-door";
        let line = |rest: &str| format!("{h} — {rest}");
        assert_eq!(
            standing_of(&line("done. Its answer:\nfixed"), h),
            Standing::Done
        );
        assert_eq!(
            standing_of(&line("asks Claude (reply with …):\nWhich?"), h),
            Standing::Asking
        );
        assert_eq!(
            standing_of(&line("stuck: the repeat guard…"), h),
            Standing::Stuck
        );
        assert_eq!(
            standing_of(&line("stopped, as you asked"), h),
            Standing::Stuck
        );
        assert_eq!(standing_of(&line("error: lost"), h), Standing::Failed);
        assert_eq!(
            standing_of(&line("round 3, reading files"), h),
            Standing::Running
        );
    }

    #[test]
    fn news_is_cut_on_a_character() {
        assert_eq!(news("  done  "), "done");
        let long = "é".repeat(NEWS_CHARS + 5);
        let cut = news(&long);
        assert_eq!(cut.chars().count(), NEWS_CHARS + 1);
        assert!(cut.ends_with('…'));
        assert_ne!(digest("a"), digest("b"));
    }
}
