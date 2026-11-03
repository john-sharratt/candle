//! `repo_job_log {job, mode, …}`: read a job's log where the agent keeps it
//! (`docs/build_deploy_tools.md` §3) — a summary, a tail, a range of lines or
//! a search over the whole log, every line numbered as it stands in the log —
//! so a caller reads a megabyte test log in a few hundred bytes.

use cluster::log_read::{Line, LogAsk, LogRead, MAX_CONTEXT, MAX_LINES};
use serde_json::{json, Value};

use super::{host_of, lost, state, Handle, NO_AGENTS};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "repo_job_log";

/// The lines a tail reads when it names no `lines`.
const DEFAULT_TAIL: u32 = 40;

/// The modes, in the order the schema offers them.
pub const MODES: [&str; 4] = ["summary", "tail", "lines", "grep"];

/// The schema a log tool takes besides what names the log: the mode first,
/// then what each mode needs, which the grammar writes after the mode that
/// needs it (`docs/stencil_tree.md` §12.1).
pub fn log_properties() -> Value {
    json!({
        "mode": {"type": "string", "enum": MODES, "description": "How to read it. `summary`: cargo's test results, the failures and errors with their line numbers, counts, and the last lines — read this first. `tail`: the last `lines`. `lines`: `count` lines from line `from` — what a summary or a search pointed at. `grep`: the lines holding `pattern`, ignoring case, with `context` lines around each."},
        "pattern": {"type": "string", "minLength": 1, "maxLength": 200, "description": "`grep`: the text to find, ignoring case; plain text, not a regular expression."},
        "context": {"type": "integer", "minimum": 0, "maximum": MAX_CONTEXT, "description": "`grep`: lines to show either side of each hit, 0 to 5; 0 by default."},
        "from": {"type": "integer", "minimum": 1, "description": "`lines`: the first line, counting from 1. `grep`: search from this line on — what a capped search says to go on from."},
        "count": {"type": "integer", "minimum": 1, "maximum": MAX_LINES, "description": "`lines`: how many lines, at most 200."},
        "lines": {"type": "integer", "minimum": 1, "maximum": MAX_LINES, "description": "`tail`: how many of the last lines, at most 200; 40 by default."}
    })
}

/// What each mode requires of a call.
pub fn log_rules() -> Value {
    json!([
        {"if": {"properties": {"mode": {"const": "grep"}}}, "then": {"required": ["pattern"]}},
        {"if": {"properties": {"mode": {"const": "lines"}}}, "then": {"required": ["from", "count"]}}
    ])
}

pub fn definition() -> Value {
    let mut properties = json!({
        "job": {"type": "string", "minLength": 18, "maxLength": 64, "description": "The job's handle, `<node>:<id>`, as `repo_job_start`, `service_deploy` or `repo_job_list` answered it."}
    });
    let fields = log_properties();
    for (k, v) in fields.as_object().expect("an object") {
        properties[k] = v.clone();
    }
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read what a build-agent job printed, read on the agent so only the answer comes \
            back — never the whole log. Every line is numbered as it stands in the log, so a line one read \
            names is the line another reads. Start with `mode: summary`: whether the tests passed (cargo's \
            `test result:` lines), the failures and compiler errors with their line numbers, how many \
            warnings, and the last lines. Then read around a numbered line with `mode: lines` (`from`, \
            `count`), search the whole log with `mode: grep` (`pattern`, plain text, ignoring case; \
            `context` lines around each hit; past 200 hits it says how many more and the `from` to go on \
            with), or read the end with `mode: tail`. Each answer holds at most 200 lines. Works while the \
            job runs — it reads what has been printed so far — and after it ends. A deploy's build is a job: \
            its handle reads here too. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["job", "mode"],
            "allOf": log_rules(),
            "additionalProperties": false
        }
    })
}

/// The ask `args` makes of a log.
pub fn ask_of(args: &Value) -> Result<LogAsk, String> {
    let n = |key: &str| args.get(key).and_then(Value::as_u64);
    let mode = string(args, "mode")?;
    match mode {
        "summary" => Ok(LogAsk::Summary),
        "tail" => Ok(LogAsk::Tail {
            lines: n("lines").map_or(DEFAULT_TAIL, |l| l.clamp(1, u64::from(MAX_LINES)) as u32),
        }),
        "lines" => Ok(LogAsk::Lines {
            from: n("from").ok_or("`mode: lines` reads from line `from`, counting from 1")?,
            count: n("count")
                .ok_or("`mode: lines` reads `count` lines, at most 200")?
                .clamp(1, u64::from(MAX_LINES)) as u32,
        }),
        "grep" => Ok(LogAsk::Grep {
            pattern: string(args, "pattern")
                .map_err(|_| "`mode: grep` searches for `pattern`, a non-empty text")?
                .to_string(),
            context: n("context").unwrap_or(0).min(u64::from(MAX_CONTEXT)) as u32,
            from: n("from").unwrap_or(1).max(1),
        }),
        other => Err(format!(
            "`{other}` is not a mode: a log is read as `summary`, `tail`, `lines` or `grep`"
        )),
    }
}

/// A log read as an answer: what was read, of how much, then the lines, and
/// how to read on. `what` names the log and `reread` is the call that reads
/// it again, with `{args}` where the arguments to read on go.
pub fn render(
    what: &str,
    ask: &LogAsk,
    read: &LogRead,
    reread: &dyn Fn(String) -> String,
) -> String {
    let mut out = format!(
        "{what}: {} lines{}",
        read.total,
        if read.ended {
            ""
        } else {
            " so far, still being written"
        }
    );
    if let Some(s) = &read.summary {
        out.push_str(&format!(
            "; {} failures, {} errors, {} warnings",
            s.failure_count, s.error_count, s.warning_count
        ));
        let group = |out: &mut String, head: &str, lines: &[Line]| {
            if lines.is_empty() {
                return;
            }
            out.push_str(&format!("\n{head}:"));
            for l in lines {
                out.push_str(&format!("\n{}: {}", l.n, l.text));
            }
        };
        group(&mut out, "test results", &s.results);
        group(&mut out, "failures", &s.failures);
        group(&mut out, "errors", &s.errors);
        group(&mut out, "last lines", &s.last);
        if let Some(n) = s
            .errors
            .first()
            .into_iter()
            .chain(s.failures.first())
            .map(|l| l.n)
            .min()
        {
            out.push_str(&format!(
                "\nRead around the first: {}.",
                reread(format!(
                    "mode: \"lines\", from: {}, count: 60",
                    n.saturating_sub(10).max(1)
                ))
            ));
        }
        return out;
    }
    match ask {
        LogAsk::Grep { pattern, .. } => {
            let hits = read
                .lines
                .iter()
                .filter(|l| l.text.to_lowercase().contains(&pattern.to_lowercase()))
                .count() as u64;
            out.push_str(&format!(
                "; {} lines hold `{pattern}`{}",
                hits + read.more,
                if read.more > 0 {
                    format!(", {hits} shown")
                } else {
                    String::new()
                }
            ));
        }
        LogAsk::Lines { .. } | LogAsk::Tail { .. } => {
            if let (Some(first), Some(last)) = (read.lines.first(), read.lines.last()) {
                out.push_str(&format!("; lines {}–{}", first.n, last.n));
            }
        }
        LogAsk::Summary => {}
    }
    for l in &read.lines {
        out.push_str(&format!("\n{}: {}", l.n, l.text));
    }
    if let Some(next) = read.next {
        let args = match ask {
            LogAsk::Grep {
                pattern, context, ..
            } => format!(
                "mode: \"grep\", pattern: {}, context: {context}, from: {next}",
                json!(pattern)
            ),
            LogAsk::Lines { count, .. } => {
                format!("mode: \"lines\", from: {next}, count: {count}")
            }
            _ => format!("mode: \"lines\", from: {next}, count: 200"),
        };
        out.push_str(&format!("\nRead on: {}.", reread(args)));
    }
    out
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let h = match string(args, "job").and_then(Handle::parse) {
        Ok(h) => h,
        Err(e) => return Answer::error(e),
    };
    let ask = match ask_of(args) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let at = h.at(&agents);
    let view = match agents.view(at, &h.id).await {
        Ok(v) => v,
        Err(e) => return Answer::error(lost(&h, &e)),
    };
    let read = match agents.read_log(at, &h.id, &ask).await {
        Ok(r) => r,
        Err(e) => return Answer::error(lost(&h, &e)),
    };
    let host = host_of(&agents, h.node).await;
    let what = format!(
        "job {h} on {host} ({})",
        if view.status.is_finished() {
            state(&view.status)
        } else {
            "running".to_string()
        }
    );
    let reread = |rest: String| format!("`repo_job_log {{job: \"{h}\", {rest}}}`");
    Answer::ok(render(&what, &ask, &read, &reread))
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use cluster::log_read::Summary;

    use super::*;

    fn line(n: u64, text: &str) -> Line {
        Line {
            n,
            text: text.into(),
        }
    }

    fn reread(rest: String) -> String {
        format!("`repo_job_log {{job: \"h\", {rest}}}`")
    }

    #[test]
    fn each_mode_asks_for_what_it_needs() {
        assert_eq!(ask_of(&json!({"mode": "summary"})), Ok(LogAsk::Summary));
        assert_eq!(
            ask_of(&json!({"mode": "tail"})),
            Ok(LogAsk::Tail {
                lines: DEFAULT_TAIL
            })
        );
        assert_eq!(
            ask_of(&json!({"mode": "lines", "from": 40, "count": 900})),
            Ok(LogAsk::Lines {
                from: 40,
                count: MAX_LINES
            })
        );
        assert_eq!(
            ask_of(&json!({"mode": "grep", "pattern": "FAILED", "context": 9})),
            Ok(LogAsk::Grep {
                pattern: "FAILED".into(),
                context: MAX_CONTEXT,
                from: 1
            })
        );
        assert!(ask_of(&json!({"mode": "lines", "from": 4}))
            .unwrap_err()
            .contains("`count`"));
        assert!(ask_of(&json!({"mode": "grep"}))
            .unwrap_err()
            .contains("`pattern`"));
        assert!(ask_of(&json!({"mode": "page"}))
            .unwrap_err()
            .contains("is not a mode"));
    }

    /// **A summary reads as its counts and its numbered lines, and points at
    /// the first failure.**
    #[test]
    fn a_summary_points_at_the_first_failure() {
        let read = LogRead {
            total: 900,
            ended: true,
            summary: Some(Summary {
                results: vec![line(880, "test result: FAILED. 1 passed; 1 failed")],
                failures: vec![line(850, "test x ... FAILED")],
                errors: Vec::new(),
                failure_count: 1,
                error_count: 0,
                warning_count: 2,
                last: vec![line(900, "error: test failed")],
            }),
            ..LogRead::default()
        };
        assert_eq!(
            render(
                "job h on BlackWorld (exited 101)",
                &LogAsk::Summary,
                &read,
                &reread
            ),
            "job h on BlackWorld (exited 101): 900 lines; 1 failures, 0 errors, 2 warnings\ntest \
             results:\n880: test result: FAILED. 1 passed; 1 failed\nfailures:\n850: test x ... \
             FAILED\nlast lines:\n900: error: test failed\nRead around the first: `repo_job_log \
             {job: \"h\", mode: \"lines\", from: 840, count: 60}`."
        );
    }

    /// **A capped search says how many lines held the pattern and the call
    /// that goes on**, its pattern quoted as JSON.
    #[test]
    fn a_capped_search_says_how_to_go_on() {
        let read = LogRead {
            total: 5_000,
            lines: vec![line(10, "warning: a"), line(11, "  --> b.rs")],
            more: 7,
            next: Some(400),
            ..LogRead::default()
        };
        let ask = LogAsk::Grep {
            pattern: "Warning".into(),
            context: 1,
            from: 1,
        };
        assert_eq!(
            render("job h", &ask, &read, &reread),
            "job h: 5000 lines so far, still being written; 8 lines hold `Warning`, 1 shown\n10: \
             warning: a\n11:   --> b.rs\nRead on: `repo_job_log {job: \"h\", mode: \"grep\", \
             pattern: \"Warning\", context: 1, from: 400}`."
        );
    }

    #[test]
    fn a_range_says_which_lines_it_holds() {
        let read = LogRead {
            total: 3,
            ended: true,
            lines: vec![line(2, "b"), line(3, "c")],
            ..LogRead::default()
        };
        assert_eq!(
            render("job h", &LogAsk::Tail { lines: 2 }, &read, &reread),
            "job h: 3 lines; lines 2–3\n2: b\n3: c"
        );
    }

    /// **The schema writes the mode before what it needs, and says what each
    /// mode requires**, so the grammar cannot close a `grep` without its
    /// pattern.
    #[test]
    fn the_schema_puts_the_mode_first_and_its_needs_after() {
        let d = definition();
        let props: Vec<&String> = d["inputSchema"]["properties"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        assert_eq!(props[..2], ["job", "mode"]);
        assert_eq!(d["inputSchema"]["required"], json!(["job", "mode"]));
        assert_eq!(
            d["inputSchema"]["allOf"][0]["then"]["required"],
            json!(["pattern"])
        );
    }
}
