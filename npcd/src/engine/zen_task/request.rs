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
                "name": slug(&text("title")?).ok_or("Say the `title` in words.")?,
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

/// The longest task name zend takes (`zend::mcp::names::MAX_NAME`).
pub const MAX_TITLE: usize = 40;

/// `title` as the name zend takes: a kebab-case slug of at most
/// [`MAX_TITLE`] characters — lowercase letters and digits, every other run
/// of characters one hyphen, cut on a word. `None` when it holds no letter or
/// digit.
pub fn slug(title: &str) -> Option<String> {
    let mut out = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    if out.len() > MAX_TITLE {
        let cut = &out[..=MAX_TITLE];
        out = match cut.rfind('-') {
            Some(at) if at > 0 => cut[..at].to_string(),
            _ => out[..MAX_TITLE].to_string(),
        };
    }
    let out = out.trim_end_matches('-').to_string();
    (!out.is_empty()).then_some(out)
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
///
/// zend opens each job's block with `{job}: {state}`
/// (`zend::mcp::tools::job::source::task::standing`): `running — …` while it
/// works, then `question`, `stuck`, `done` or `error`. A task zend no longer
/// holds is `{job}: there is no task …`.
pub fn standing_of(text: &str, handle: &str) -> Standing {
    let Some(state) = state_of(text, handle) else {
        return Standing::Running;
    };
    if state.starts_with("done") {
        Standing::Done
    } else if state.starts_with("question") {
        Standing::Asking
    } else if state.starts_with("stuck") {
        Standing::Stuck
    } else if state.starts_with("error") || state.starts_with("there is no task") {
        Standing::Failed
    } else {
        Standing::Running
    }
}

/// What follows `{handle}: ` on the line that opens `handle`'s block.
fn state_of<'t>(text: &'t str, handle: &str) -> Option<&'t str> {
    let opener = format!("{handle}: ");
    text.lines()
        .find_map(|l| l.strip_prefix(opener.as_str()))
        .map(str::trim)
}

/// What `job_status` said of `handle` that is news to its carrier: the state
/// its block opens with and what follows it — the question, the answer, the
/// journal entries — without what a running task's lines say of its turn: what
/// it is doing, for how long and with how many tokens changes on every look,
/// and is not news. The block's closing `pages:` line is not news either.
pub fn news_of(text: &str, handle: &str) -> String {
    let opener = format!("{handle}: ");
    let mut lines = text.lines().skip_while(|l| !l.starts_with(opener.as_str()));
    let Some(first) = lines.next() else {
        return text.trim().to_string();
    };
    let state = first[opener.len()..].trim();
    let running = state.starts_with("running");
    let head = if running { "running" } else { state };
    let turn = format!("{handle} — ");
    let body: Vec<&str> = lines
        .take_while(|l| !l.starts_with("pages: "))
        .filter(|l| !(running && l.starts_with(turn.as_str())))
        .collect();
    format!("{handle}: {head}\n{}", body.join("\n"))
        .trim()
        .to_string()
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
                &args(json!({"title": "fix-door", "prompt": "Fix the door.", "role": "coder"})),
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
            &args(json!({"title": "a", "prompt": "b"})),
            &t,
            h
        )
        .unwrap_err()
        .contains("started already"));
        assert_eq!(
            zend_call("task_start", &args(json!({"title": "a"})), &t, None).unwrap_err(),
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

    const H: &str = "task:mission-7-3/fix-door";

    /// One job's block as zend's `job_status` renders it
    /// (`zend::mcp::tools::job::status::render` over `source::task::standing`
    /// and `tools::wait::result_of`): `{job}: {state}`, the result under it
    /// opening `{job} — …`, and the pages line.
    fn block(state: &str, result: &str) -> String {
        format!(
            "{H}: {state}\n{H} — {result}\npages: 2 — `job_read {{job: \"{H}\", page: P}}` reads page P"
        )
    }

    fn running(secs: u32, tokens: &str) -> String {
        let doing = format!(
            "running {secs}s, 3 rounds: {tokens} thinking tokens, 4 tool calls (file_read 4); now thinking"
        );
        block(&format!("running — {doing}"), &doing)
    }

    /// **Where a task stands is read from the line its block opens with**,
    /// each state by zend's word for it; a task zend no longer holds failed.
    #[test]
    fn a_tasks_standing_is_read_from_its_line() {
        assert_eq!(
            standing_of(&block("done", "done. Its answer:\nfixed"), H),
            Standing::Done
        );
        assert_eq!(
            standing_of(
                &block(
                    "question",
                    "asks Claude (reply with `repo_task_reply`; it waits until then):\nWhich?"
                ),
                H
            ),
            Standing::Asking
        );
        assert_eq!(
            standing_of(
                &block(
                    "stuck",
                    "stuck: the repeat guard or the round cap closed its tools."
                ),
                H
            ),
            Standing::Stuck
        );
        assert_eq!(
            standing_of(&block("error", "error: the worker was lost"), H),
            Standing::Failed
        );
        assert_eq!(standing_of(&running(30, "1,234"), H), Standing::Running);
        assert_eq!(
            standing_of(&format!("{H}: there is no task `{H}`"), H),
            Standing::Failed
        );
        assert_eq!(
            standing_of("task:mission-7-3/other: done", H),
            Standing::Running,
            "another task's line is not this one's"
        );
    }

    /// **A running task's news does not change as it runs**: its elapsed time
    /// and tokens are left out, so the same news is posted once, and what it
    /// says once it settles is news.
    #[test]
    fn a_running_tasks_news_holds_still_between_looks() {
        let first = news_of(&running(30, "1,234"), H);
        assert_eq!(first, format!("{H}: running"));
        assert_eq!(news_of(&running(45, "2,001"), H), first);
        let done = news_of(&block("done", "done. Its answer:\nfixed"), H);
        assert_eq!(done, format!("{H}: done\n{H} — done. Its answer:\nfixed"));
        assert_ne!(digest(&done), digest(&first));
    }

    /// **A title becomes the name zend takes**: a kebab-case slug of at most
    /// 40 characters, cut on a word; one with no letter or digit is none.
    #[test]
    fn a_title_becomes_a_kebab_slug() {
        assert_eq!(
            slug("Fix the door sensor build!").as_deref(),
            Some("fix-the-door-sensor-build")
        );
        assert_eq!(slug("fix-door").as_deref(), Some("fix-door"));
        assert_eq!(slug("  --Door__Sensor--  ").as_deref(), Some("door-sensor"));
        assert_eq!(
            slug("Replace the broken east door sensor and rebuild the firmware").as_deref(),
            Some("replace-the-broken-east-door-sensor-and")
        );
        assert_eq!(slug("x".repeat(50).as_str()).map(|s| s.len()), Some(40));
        assert_eq!(slug("!!!"), None);
        assert_eq!(
            zend_call(
                "task_start",
                &args(json!({"title": "Fix the Door", "prompt": "p"})),
                "t",
                None
            )
            .unwrap()
            .1["name"],
            "fix-the-door"
        );
        assert_eq!(
            zend_call(
                "task_start",
                &args(json!({"title": "…", "prompt": "p"})),
                "t",
                None
            )
            .unwrap_err(),
            "Say the `title` in words."
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
